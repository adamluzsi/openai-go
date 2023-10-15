package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/adamluzsi/frameless/pkg/enum"
	"github.com/adamluzsi/frameless/pkg/errorkit"
	"github.com/adamluzsi/frameless/pkg/httpkit"
	"github.com/adamluzsi/frameless/pkg/logger"
	"github.com/adamluzsi/frameless/pkg/retry"
	"github.com/adamluzsi/frameless/pkg/zerokit"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

var globlock sync.RWMutex

const defaultBaseURL = "https://api.openai.com"

// ChatModelID represents the ID of the chat model to use.
type (
	ChatModelID     string
	InstructModelID string
)

// Various ChatModelID constants.
const (
	GPT4     ChatModelID = "gpt-4"             // GPT-4 model
	GPT4_32k ChatModelID = "gpt-4-32k"         // GPT-4 model with 32k tokens
	GPT3     ChatModelID = "gpt-3.5-turbo"     // GPT-3.5 Turbo model
	GPT3_16k ChatModelID = "gpt-3.5-turbo-16k" // GPT-3.5 Turbo model with 16k tokens

	// Babbage is a GPT base models, which is not optimized for instruction-following
	// and are less capable, but they can be effective when fine-tuned for narrow tasks.
	// They also cost-efficient to use for testing purposes.
	// The Babbage model usage cost is $0.0004 / 1K tokens.
	Babbage InstructModelID = "babbage-002"
)

// Client represents the OpenAI API client.
type Client struct {
	BaseURL       string
	APIKey        string
	HTTPClient    *http.Client
	RetryStrategy retry.Strategy

	onInit sync.Once

	functionMapping map[ChatFunctionName]ChatFunctionMapping
}

var DefaultRetryStrategy retry.Strategy = retry.Jitter{
	MaxRetries:      9,
	MaxWaitDuration: 3 * time.Second,
}

func (c *Client) Init() {
	c.onInit.Do(func() {
		if c.BaseURL == "" {
			c.BaseURL = defaultBaseURL
		}
		if c.HTTPClient == nil {
			// GPT4 is very slow, a long timeout sadly is necessary
			c.HTTPClient = &http.Client{Timeout: 5 * time.Minute}
		}
		if c.RetryStrategy == nil {
			c.RetryStrategy = DefaultRetryStrategy
		}
		c.HTTPClient.Transport = httpkit.RetryRoundTripper{
			Transport: zerokit.Coalesce[http.RoundTripper](
				c.HTTPClient.Transport, http.DefaultTransport),
			RetryStrategy: c.RetryStrategy,
		}
	})
}

// getAPIKey returns the API key for authentication.
func (c *Client) getAPIKey() string {
	if c.APIKey == "" {
		if apiKey, ok := os.LookupEnv("OPENAI_API_KEY"); ok {
			c.APIKey = apiKey
		}
	}
	return c.APIKey
}

func (c *Client) getHTTPClient() *http.Client {
	c.Init()
	return c.HTTPClient
}

type ChatSession struct {
	// Model specifies the ID of the model to use (e.g., "text-davinci-002").
	Model ChatModelID `json:"model"`

	// Messages is an array of ChatMessage structs representing the conversation history.
	Messages []ChatMessage `json:"messages"`

	// Functions is an array of Function objects that describe the functions
	// available for the GPT model to call during the chat completion.
	// Each Function object should contain details like the function's name,
	// description, and parameters. The GPT model will use this information
	// to decide whether to call a function based on the user's query.
	// For example, you can define functions like send_email(to: string, body: string)
	// or get_current_weather(location: string, unit: 'celsius' | 'fahrenheit').
	// Note: Defining functions will count against the model's token limit.
	Functions []ChatFunction
}

func (session ChatSession) Clone() ChatSession {
	msgsCopy := make([]ChatMessage, len(session.Messages))
	copy(msgsCopy, session.Messages)
	session.Messages = msgsCopy
	return session
}

func (session ChatSession) withMessage(role ChatMessageRole, content string) ChatSession {
	session = session.Clone()
	session.Messages = append(session.Messages, ChatMessage{Role: role, Content: content})
	return session
}
func (session ChatSession) WithSystemMessage(content string) ChatSession {
	return session.withMessage(SystemChatMessageRole, content)
}

func (session ChatSession) WithUserMessage(content string) ChatSession {
	return session.withMessage(UserChatMessageRole, content)
}

func (session ChatSession) WithAssistantMessage(content string) ChatSession {
	return session.withMessage(AssistantChatMessageRole, content)
}

func (session ChatSession) LastAssistantContent() string {
	for i := len(session.Messages) - 1; i >= 0; i-- {
		msg := session.Messages[i]
		if msg.Role == AssistantChatMessageRole {
			return msg.Content
		}
	}
	return ""
}

func (session ChatSession) LookupLastMessage() (ChatMessage, bool) {
	if len(session.Messages) == 0 {
		return ChatMessage{}, false
	}
	return session.Messages[len(session.Messages)-1], true
}

// ChatCompletionRequest represents the parameters for a chat completion request.
type ChatCompletionRequest struct {
	// Model specifies the ID of the model to use (e.g., "text-davinci-002").
	Model ChatModelID `json:"model"`

	// Messages is an array of ChatMessage structs representing the conversation history.
	Messages []ChatMessage `json:"messages"`

	// MaxTokens specifies the maximum number of tokens for the message output.
	// Optional; if not provided, the API will use the model's maximum limit.
	MaxTokens *int `json:"max_tokens,omitempty"`

	// Temperature controls the randomness of the output, ranging from 0.0 to 1.0.
	// Optional; if not provided, the API will use a default value.
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP controls the diversity of the output via nucleus sampling, ranging from 0.0 to 1.0.
	// Optional; if not provided, the API will use a default value.
	TopP *float64 `json:"top_p,omitempty"`

	// FrequencyPenalty alters the likelihood of tokens appearing based on their frequency, ranging from -2.0 to 2.0.
	// Optional; if not provided, the API will use a default value.
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`

	// PresencePenalty alters the likelihood of tokens appearing based on their presence in the prompt, ranging from -2.0 to 2.0.
	// Optional; if not provided, the API will use a default value.
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`

	// StopSequences is an array of strings that indicate the end of the generated content.
	// Optional; if not provided, the API will decide when to stop the output.
	StopSequences []string `json:"stop_sequences,omitempty"`

	// UserID is an optional identifier for the user, used for fine-tuned models.
	// Optional; if not provided, the API will not personalize the response.
	UserID string `json:"user,omitempty"`

	// SessionID is an optional identifier for the session, used for fine-tuned models.
	// Optional; if not provided, the API will not maintain context between API calls.
	SessionID string `json:"session_id,omitempty"`

	// Functions is an array of Function objects that describe the functions
	// available for the GPT model to call during the chat completion.
	// Each Function object should contain details like the function's name,
	// description, and parameters. The GPT model will use this information
	// to decide whether to call a function based on the user's query.
	// For example, you can define functions like send_email(to: string, body: string)
	// or get_current_weather(location: string, unit: 'celsius' | 'fahrenheit').
	// Note: Defining functions will count against the model's token limit.
	Functions []ChatFunction `json:"functions,omitempty"`

	// FunctionCall is a string that specifies the behavior of function calling
	// during the chat completion. It can have the following values:
	// - "auto": The model decides whether to call a function and which function to call.
	// - "none": Forces the model to not call any function.
	// - { "name": "<function_name>" }: Forces the model to call a specific function by name.
	// This field allows you to control the model's decision-making process regarding
	// function calls, providing a way to either automate or manually control actions.
	// When left nil, it is interpreted as "auto" on OpenAI API side.
	FunctionCall *ChatFunctionCall `json:"function_call,omitempty"`
}

// cloneMessages makes it safe to manipulate the []ChatMessages list in the ChatCompletionRequest.
func (ccr *ChatCompletionRequest) cloneMessages() {
	msgs := make([]ChatMessage, len(ccr.Messages))
	copy(msgs, ccr.Messages)
	ccr.Messages = msgs
}

type ChatMessageRole string

const (
	// SystemChatMessageRole is a prompt meant to instrument the Assistant.
	SystemChatMessageRole ChatMessageRole = "system"
	// UserChatMessageRole is a user prompt input.
	UserChatMessageRole ChatMessageRole = "user"
	// AssistantChatMessageRole is a reply type.
	// GPT for example uses the AssistantChatMessageRole to reply back to its caller.
	AssistantChatMessageRole ChatMessageRole = "assistant"
	// FunctionChatMessageRole is used to respond back
	// to a function call request by the AssistantChatMessageRole.
	FunctionChatMessageRole ChatMessageRole = "function"
)

// ChatMessage represents a single message in the conversation history.
type ChatMessage struct {
	// Role specifies the role of the message sender, usually "system", "user", or "assistant".
	Role ChatMessageRole `json:"role" enum:"system;user;assistant;"`

	// Content contains the actual text of the message.
	Content string `json:"content"`

	// FunctionName is required when Role is FunctionChatMessageRole.
	FunctionName ChatFunctionName `json:"name,omitempty"`
	// FunctionCall is populated in a response when GPT requires a function execution.
	FunctionCall *ChatFunctionCall `json:"function_call,omitempty"`
}

// ChatCompletionResponse represents the response from a chat completion request.
type ChatCompletionResponse struct {
	// ID is a unique identifier for the chat completion.
	// Example: "chatcmpl-abc123"
	ID string `json:"id"`

	// Object is the object type, always "chat.completion".
	// Example: "chat.completion"
	Object string `json:"object" enum:"chat.completion;"`

	// Created is the Unix timestamp (in seconds) of when the chat completion was created.
	// Example: 1677858242
	Created int `json:"created"`

	// Model is the model used for the chat completion.
	// Example: "gpt-3.5-turbo-0613"
	Model ChatModelID `json:"model"`

	// Usage contains usage statistics for the completion request.
	// Example: {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}
	Usage ChatCompletionUsage `json:"usage"`

	// Choices is a list of chat completion choices. Can be more than one if 'n' is greater than 1.
	// Example: [{"message": {"role": "assistant", "content": "This is a test!"}, "finish_reason": "stop", "index": 0}]
	Choices []Choice `json:"choices"`
}

type ChatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Choice struct {
	// Message contains the role and content of the message.
	// Example: {"role": "assistant", "content": "This is a test!"}
	Message ChatMessage `json:"message"`

	// FinishReason is the reason the API stopped generating further tokens.
	// Common values are "stop", "length", etc.
	// Example: "stop"
	FinishReason FinishReason `json:"finish_reason"`

	// Index is the index of the choice in the array.
	// Example: 0
	Index int `json:"index"`
}

func (c *Client) ChatSession(ctx context.Context, session ChatSession) (ChatSession, error) {
	c.Init()

	for _, fn := range session.Functions {
		if err := fn.Validate(); err != nil {
			return ChatSession{}, err
		}
	}

	// Create a deep copy of the original ChatSession's Messages to avoid modifying it
	session = session.Clone()

call:
	response, err := c.ChatCompletion(ctx, ChatCompletionRequest{
		Model:     session.Model,
		Messages:  session.Messages,
		Functions: session.Functions,
	})

	if errors.Is(err, ErrContextLengthExceeded) {
		switch session.Model {
		case GPT3:
			session.Model = GPT3_16k
			goto call
		case GPT4:
			session.Model = GPT4_32k
			goto call
		}
	}

	if err != nil {
		return ChatSession{}, err
	}

	for _, choice := range response.Choices {
		session.Messages = append(session.Messages, choice.Message)
	}

	if 0 < len(response.Choices) {
		lastChoice := response.Choices[len(response.Choices)-1]

		if lastChoice.FinishReason == FinishReasonFunctionCall &&
			lastChoice.Message.FunctionCall != nil {
			fnCall := *lastChoice.Message.FunctionCall

			for _, fn := range session.Functions {
				if fn.Name == fnCall.Name && fn.Exec != nil {

					result, fnErr := fn.Exec(ctx, json.RawMessage(fnCall.Arguments))
					if fnErr != nil { // if the function encountered an error, GPT needs to know about it.
						logger.Warn(ctx, "function execution encountered an error",
							logger.Field("functionName", fn.Name),
							logger.ErrField(fnErr))
						message, err := MakeFunctionChatMessage(fn.Name, functionErrorContent{
							Error: fnErr.Error(),
						})
						if err != nil { // if the encoding had an error, the developer has to know about it
							return session, err
						}
						session.Messages = append(session.Messages, message)
						goto call
					}

					message, err := MakeFunctionChatMessage(fn.Name, result)
					if err != nil { // if the encoding had an error, the developer has to know about it
						return session, err
					}
					session.Messages = append(session.Messages, message)
					goto call
				}
			}
		}
	}

	return session, nil
}

func (c *Client) ChatCompletion(ctx context.Context, cc ChatCompletionRequest) (ChatCompletionResponse, error) {
	c.Init()

	if cc.Functions != nil {
		cc.Messages = append([]ChatMessage{fixFunctionHallucinationMessage}, cc.Messages...)

		for i, fn := range cc.Functions {
			var required = make(map[string]struct{})
			for _, req := range fn.Parameters.Required {
				required[req] = struct{}{}
			}
			for name, prop := range fn.Parameters.Properties {
				if prop.Required {
					required[name] = struct{}{}
				}
			}
			var requiredProperties []string
			for prop := range required {
				requiredProperties = append(requiredProperties, prop)
			}
			cc.Functions[i].Parameters.Required = requiredProperties
		}
	}

	var response ChatCompletionResponse
	jsonPayload, err := json.Marshal(cc)
	if err != nil {
		return response, err
	}

	uri := c.BaseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", uri, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return response, err
	}

	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	req.Header.Set("Authorization", "Bearer "+c.getAPIKey())
	req.Header.Set("Accept", "application/json; charset=utf-8")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return response, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return response, err
	}

	if resp.StatusCode != http.StatusOK {
		var errResp errorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return response, err
		}
		if errResp.Error.Code == errorCodeContextLengthExceeded {
			return response, fmt.Errorf("%w: %s",
				ErrContextLengthExceeded,
				errResp.Error.Message)
		}
		return response, fmt.Errorf("%s: %s",
			errResp.Error.Code, errResp.Error.Message)
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return response, err
	}

	return response, nil
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

const errorCodeContextLengthExceeded = "context_length_exceeded"

const ErrContextLengthExceeded errorkit.Error = "ErrContextLengthExceeded"

// Chat Function

type JSONSchema struct {
	Type       string                        `json:"type"`
	Properties map[string]JSONSchemaProperty `json:"properties"`
	Items      *JSONSchemaItems              `json:"items,omitempty"`
	// Required mark which property is required.
	// It gets autopopulated from the properties flagged as "Required"
	Required []string `json:"required,omitempty"`
}

type JSONSchemaProperty struct {
	Type        string           `json:"type"`
	Description string           `json:"description"`
	Enum        []string         `json:"enum,omitempty"`
	Items       *JSONSchemaItems `json:"items,omitempty"`
	Required    bool             `json:"-"`
}

type JSONSchemaItems struct {
	// Type specifies the data type of the elements. Common types include "string", "number", "integer", "boolean", "array", and "object".
	Type string `json:"type"`
	// Enum specifies the set of allowed values for the elements.
	Enum []string `json:"enum,omitempty"`
}

type ChatFunction struct {
	Name        ChatFunctionName `json:"name"`
	Description string           `json:"description"`
	Parameters  JSONSchema       `json:"parameters"`
	Exec        ChatFunctionExec `json:"-"`
}

type ChatFunctionExec func(ctx context.Context, payload json.RawMessage) (any, error)

func (cfn ChatFunction) Validate() error {
	if cfn.Exec == nil {
		return ErrFunctionMissingExec
	}
	return nil
}

const ErrFunctionMissingExec errorkit.Error = "ErrFunctionMissingExec: Your function declaration is missing the execution function"

type ChatFunctionName string

// ChatFunctionCall is the request that the Assistant asks from us to complete.
type ChatFunctionCall struct {
	Name ChatFunctionName `json:"name,omitempty"`
	// Arguments is a JSON encoded call function with arguments in JSON format
	Arguments string `json:"arguments,omitempty"`
}

const FixFunctionHallucination = "Only use the functions you have been provided with." // System

var fixFunctionHallucinationMessage = ChatMessage{
	Role:    SystemChatMessageRole,
	Content: FixFunctionHallucination,
}

func MakeFunctionChatMessage(name ChatFunctionName, contentDTO any) (ChatMessage, error) {
	data, err := json.Marshal(contentDTO)
	if err != nil {
		return ChatMessage{}, err
	}
	return ChatMessage{
		Role:         FunctionChatMessageRole,
		Content:      string(data),
		FunctionName: name,
	}, nil
}

type functionErrorContent struct {
	Error string `json:"error"`
}

type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonFunctionCall  FinishReason = "function_call"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonNull          FinishReason = "null"
)

var _ = enum.Register[FinishReason](
	FinishReasonStop,
	FinishReasonLength,
	FinishReasonFunctionCall,
	FinishReasonContentFilter,
	FinishReasonNull,
)

type ChatFunctionMapping interface {
	GetParameters() JSONSchema
	Call(ChatFunctionCall) (ChatMessage, error)
}

type chatFunctionMapping[Fn any] struct {
	getParameters func() JSONSchema
	callFunc      func(ChatFunctionCall) func(ChatMessage, error)
}
