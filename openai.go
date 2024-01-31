package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go.llib.dev/frameless/pkg/enum"
	"go.llib.dev/frameless/pkg/errorkit"
	"go.llib.dev/frameless/pkg/httpkit"
	"go.llib.dev/frameless/pkg/logger"
	"go.llib.dev/frameless/pkg/pointer"
	"go.llib.dev/frameless/pkg/retry"
	"go.llib.dev/frameless/pkg/zerokit"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

const defaultBaseURL = "https://api.openai.com"

// ChatModelID represents the ID of the chat model to use.
type (
	ChatModelID     string
	InstructModelID string
)

const ( // GPT4
	GPT4         ChatModelID = GPT4_Preview
	GPT4_Vision  ChatModelID = "gpt-4-vision-preview"
	GPT4_Preview ChatModelID = "gpt-4-1106-preview"
	GPT4_Stable  ChatModelID = "gpt-4"
	// GPT4_32k
	// DEPRECATED: use GPT4
	GPT4_32k ChatModelID = "gpt-4-32k" // GPT-4 model with 32k tokens
)

const ( // GPT3
	GPT3 ChatModelID = "gpt-3.5-turbo" // GPT-3.5 Turbo model

	// GPT3_16k
	// DEPRECATED: use GPT3 directly
	GPT3_16k ChatModelID = "gpt-3.5-turbo-16k" // GPT-3.5 Turbo model with 16k tokens
)

// Babbage is a GPT base models, which is not optimized for instruction-following
// and are less capable, but they can be effective when fine-tuned for narrow tasks.
// They also cost-efficient to use for testing purposes.
// The Babbage model usage cost is $0.0004 / 1K tokens.
const Babbage InstructModelID = "babbage-002"

// Client represents the OpenAI API client.
type Client struct {
	BaseURL       string
	APIKey        string
	HTTPClient    *http.Client
	RetryStrategy retry.Strategy[retry.FailureCount]

	onInit sync.Once

	functionMapping map[FunctionName]FunctionMapping
}

var DefaultRetryStrategy retry.Strategy[retry.FailureCount] = retry.ExponentialBackoff{
	MaxRetries:      16,
	BackoffDuration: time.Second,
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

type ChatSession struct{ ChatCompletion }

func (session ChatSession) Clone() ChatSession {
	session.ChatCompletion = session.ChatCompletion.Clone()
	return session
}

func (session ChatSession) WithModel(model ChatModelID) ChatSession {
	session.Model = model
	return session
}

func (session ChatSession) WithMessage(msgs ...ChatMessage) ChatSession {
	session.Messages = append(append([]ChatMessage{}, session.Messages...), msgs...)
	return session
}

func (session ChatSession) WithSystemMessage(content string) ChatSession {
	return session.WithMessage(SystemChatMessage.From(content))
}

func (session ChatSession) WithUserMessage(content string) ChatSession {
	return session.WithMessage(UserChatMessage.From(content))
}

func (session ChatSession) WithAssistantMessage(content string) ChatSession {
	return session.WithMessage(AssistantChatMessage.From(content))
}

func (session ChatSession) LastAssistantContent() string {
	for i := len(session.Messages) - 1; i >= 0; i-- {
		msg := session.Messages[i]
		if msg.Role == AssistantChatMessage {
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

func (session ChatSession) WithFunction(fns ...Function) ChatSession {
	session.ChatCompletion.Functions = cloneSlice(session.ChatCompletion.Functions)
	session.ChatCompletion.Functions = append(session.ChatCompletion.Functions, fns...)
	return session
}

// ChatCompletion represents the parameters for a chat completion request.
type ChatCompletion struct {
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
	Functions []Function `json:"-"`

	// FunctionCall is a string that specifies the behavior of function calling
	// during the chat completion. It can have the following values:
	// - "auto": The model decides whether to call a function and which function to call.
	// - "none": Forces the model to not call any function.
	// - { "name": "<function_name>" }: Forces the model to call a specific function by name.
	// This field allows you to control the model's decision-making process regarding
	// function calls, providing a way to either automate or manually control actions.
	// When left nil, it is interpreted as "auto" on OpenAI API side.
	FunctionCall *FunctionCall `json:"function_call,omitempty"`

	// Tools is the list of enabled tooling for the assistant.
	// There can be a maximum of 128 tools per assistant.
	// Tools can be of types code_interpreter, retrieval, or function.
	//   example: [{ "type": "code_interpreter" }]
	Tools []ChatCompletionTool `json:"tools,omitempty"`

	// ToolChoice controls which (if any) function is called by the model.
	// "none" is the default when no functions are present (NoneToolChoice).
	// "auto" is the default if functions are present (AutoToolChoice).
	ToolChoice ToolChoice `json:"tool_choice,omitempty"`

	// ResponseFormat is an object specifying the format that the model must output.
	//
	// Setting to { "type": "json_object" } enables JSON mode (JSONResponseFormat),
	// which guarantees the message the model generates is valid JSON.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

func (cc ChatCompletion) MarshalJSON() ([]byte, error) {
	type DTO ChatCompletion
	var dto DTO
	dto = DTO(cc.Clone())
	if 0 < len(dto.Functions) {
		for _, fn := range dto.Functions {
			dto.Tools = append(dto.Tools, jsonMappingFunctionTool{Function: fn})
		}
		dto.Functions = nil
	}
	return json.Marshal(dto)
}

type ChatCompletionTool interface {
	GetType() ToolType
	json.Marshaler
}

func clonePointer[T any](ptr *T) *T {
	if ptr == nil {
		return ptr
	}
	v := *ptr
	return &v
}

func cloneSlice[T any](vs []T) []T {
	if vs == nil {
		return vs
	}
	nvs := make([]T, len(vs))
	copy(nvs, vs)
	return nvs
}

// cloneMessages makes it safe to manipulate the []ChatMessages list in the ChatCompletion.
func (cc *ChatCompletion) cloneMessages() {
	cc.Messages = cloneSlice(cc.Messages)
}

func (cc ChatCompletion) Clone() ChatCompletion {
	cc.cloneMessages()
	cc.MaxTokens = clonePointer(cc.MaxTokens)
	cc.Temperature = clonePointer(cc.Temperature)
	cc.FrequencyPenalty = clonePointer(cc.FrequencyPenalty)
	cc.PresencePenalty = clonePointer(cc.PresencePenalty)
	cc.Functions = cloneSlice(cc.Functions)
	cc.FunctionCall = clonePointer(cc.FunctionCall)
	cc.TopP = clonePointer(cc.TopP)
	cc.StopSequences = cloneSlice(cc.StopSequences)
	return cc
}

type ChatMessageRole string

func (cmr ChatMessageRole) From(content string) ChatMessage {
	return ChatMessage{Role: cmr, Content: content}
}

const (
	// SystemChatMessage is a prompt meant to instrument the Assistant.
	SystemChatMessage ChatMessageRole = "system"
	// UserChatMessage is a user prompt input.
	UserChatMessage ChatMessageRole = "user"
	// AssistantChatMessage is a reply type.
	// GPT for example uses the AssistantChatMessage to reply back to its caller.
	AssistantChatMessage ChatMessageRole = "assistant"
	// FunctionChatMessage is used to respond back
	// to a function call request by the AssistantChatMessage.
	// DEPRECATED: use ToolChatMessage instead.
	FunctionChatMessage ChatMessageRole = "function"
	// ToolChatMessage is used to respond back to a tool request, such as a function call.
	ToolChatMessage ChatMessageRole = "tool"
)

// ChatMessage represents a single message in the conversation history.
type ChatMessage struct {
	// Role specifies the role of the message sender, usually "system", "user", or "assistant".
	Role ChatMessageRole `json:"role" enum:"system;user;assistant;"`

	// Content contains the actual text of the message.
	Content string `json:"content"`

	// ToolCallID is the id of the Tool call that this message is responding to
	// when ChatMessage is used as part of a API request.
	ToolCallID ToolCallID `json:"tool_call_id,omitempty"`

	// The tool calls generated by the model, such as function calls.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
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
	Choices []ChatCompletionResponseChoice `json:"choices"`
}

type ChatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatCompletionResponseChoice struct {
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
	response, err := c.ChatCompletion(ctx, session.ChatCompletion)

	if errors.Is(err, ErrContextLengthExceeded) {
		switch session.Model {
		case GPT3:
			session.Model = GPT3_16k
			goto call
		case GPT3_16k:
			session.Model = GPT4_Preview
			goto call
		case GPT4_Stable:
			session.Model = GPT4_Preview
			goto call
		}
	}

	if err != nil {
		return ChatSession{}, err
	}

	for _, choice := range response.Choices {
		session.Messages = append(session.Messages, choice.Message)
	}

	if len(response.Choices) == 1 {
		var ok bool
		session, ok, err = c.doToolCalls(ctx, session, response.Choices[0])
		if err != nil {
			return session, err
		}
		if ok {
			// if fnCall was executed, we need reply to GPT about the results
			goto call
		}
	} else {
		logger.Warn(ctx, "openai.Client.doToolCalls is unable to handle multi choices")
	}

	return session, nil
}

// doToolCalls currently can't handle multi-choice situations.
func (c *Client) doToolCalls(ctx context.Context, session ChatSession, choice ChatCompletionResponseChoice) (_ ChatSession, rok bool, _ error) {
	if choice.Message.ToolCalls == nil {
		return session, false, nil
	}
	logger.Debug(ctx, "doToolCalls",
		logger.Field("tool_calls", choice.Message.ToolCalls))

	for _, toolCall := range choice.Message.ToolCalls {
		fnCall, ok := toolCall.LookupFunctionCall()
		if !ok {
			continue
		}
		var err error
		// append callFunction to the current session
		session, ok, err = c.callFunction(ctx, session, toolCall.ID, fnCall)
		if err != nil {
			return session, ok, err
		}
		rok = rok || ok
	}
	return session, rok, nil
}

func (c *Client) callFunction(ctx context.Context, session ChatSession, tcID ToolCallID, fnCall FunctionCall) (ChatSession, bool, error) {
	for _, fn := range session.Functions {
		if fn.Name != fnCall.Name {
			continue
		}
		if fn.Exec == nil { // When plugin has no Exec, automatic execution is not supported
			return session, false, nil
		}

		result, fnErr := fn.Exec(ctx, json.RawMessage(fnCall.Arguments))
		if errors.Is(fnErr, ExecInterrupt) {
			return session, false, nil
		}
		if fnErr != nil { // if the function encountered an error, GPT needs to know about it.
			logger.Warn(ctx, "function execution encountered an error",
				logger.Field("functionName", fn.Name),
				logger.ErrField(fnErr))
			message, err := MakeFunctionChatMessage(tcID, functionErrorContent{
				Error: fnErr.Error(),
			})
			if err != nil { // if the encoding had an error, the developer has to know about it
				return session, true, err
			}
			return session.WithMessage(message), true, nil
		}

		message, err := MakeFunctionChatMessage(tcID, result)
		if err != nil { // if the encoding had an error, the developer has to know about it
			return session, true, err
		}
		return session.WithMessage(message), true, nil

	}
	return session, false, nil
}

func (c *Client) ChatCompletion(ctx context.Context, cc ChatCompletion) (ChatCompletionResponse, error) {
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

	var faultCount int = -1
	uri := c.BaseURL + "/v1/chat/completions"

makeRequest:
	faultCount++
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
		if !json.Valid(body) {
			err := fmt.Errorf("%d:\n%s", resp.StatusCode, string(body))
			logger.Debug(ctx, "error occured during openai.ChatCompletion",
				logger.ErrField(err))
			return response, err
		}

		var errResp errorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return response, err
		}
		logger.Debug(ctx, "error occured during openai.ChatCompletion",
			logger.Field("error", errResp))

		switch errerr := errResp.Error; errerr.Code {
		case errCodeContextLengthExceeded:
			return response, fmt.Errorf("%w: %s",
				ErrContextLengthExceeded,
				errerr.Message)

		case errRateLimitExceeded:
			if c.RetryStrategy.ShouldTry(ctx, faultCount) {
				goto makeRequest
			}
			return response, fmt.Errorf("%w: %s",
				ErrRateLimitExceeded,
				errerr.Message)

		default:
			return response, fmt.Errorf("%s: %s",
				errerr.Code, errerr.Message)
		}
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

var _ = logger.RegisterFieldType[errorResponse](func(r errorResponse) logger.LoggingDetail {
	return logger.Fields{
		"code":    r.Error.Code,
		"message": r.Error.Message,
		"type":    r.Error.Type,
		"param":   r.Error.Param,
	}
})

const (
	errCodeContextLengthExceeded = "context_length_exceeded"
	errRateLimitExceeded         = "rate_limit_exceeded"
)

const (
	ErrContextLengthExceeded errorkit.Error = "ErrContextLengthExceeded"
	ErrRateLimitExceeded     errorkit.Error = "ErrRateLimitExceeded"
)

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

type Function struct {
	Name        FunctionName `json:"name"`
	Description string       `json:"description"`
	Parameters  JSONSchema   `json:"parameters"`
	// Exec is used to execute the Function when GPT request it.
	// If Exec is not supplied, automatic Function execution is disabled for this function.
	Exec FunctionExec `json:"-"`
}

type FunctionExec func(ctx context.Context, payload json.RawMessage) (any, error)

const ExecInterrupt errorkit.Error = "ExecInterrupt"

func (cfn Function) Validate() error {
	return nil
}

type FunctionName string

// FunctionCall is the request that the Assistant asks from us to complete.
type FunctionCall struct {
	Name FunctionName `json:"name,omitempty"`
	// Arguments is a JSON encoded call function with arguments in JSON format
	Arguments string `json:"arguments,omitempty"`
}

var _ = logger.RegisterFieldType[FunctionCall](func(call FunctionCall) logger.LoggingDetail {
	return logger.Fields{
		"name": string(call.Name),
		"args": string(call.Arguments),
	}
})

// System
const FixFunctionHallucination = `
Only use the functions you have been provided with.
You must use JSON format for the argument to make a function call. 
`

var fixFunctionHallucinationMessage = ChatMessage{
	Role:    SystemChatMessage,
	Content: FixFunctionHallucination,
}

func MakeFunctionChatMessage(tcID ToolCallID, contentDTO any) (ChatMessage, error) {
	data, err := json.Marshal(contentDTO)
	if err != nil {
		return ChatMessage{}, err
	}
	return ChatMessage{
		Role:       ToolChatMessage,
		ToolCallID: tcID,
		Content:    string(data),
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
	FinishReasonToolCalls     FinishReason = "tool_calls"
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

type FunctionMapping interface {
	GetParameters() JSONSchema
	Call(FunctionCall) (ChatMessage, error)
}

type ToolType string

const (
	CodeInterpreterToolType ToolType = "code_interpreter"
	RetrievalToolType       ToolType = "retrieval"
	FunctionToolType        ToolType = "function"
)

var _ = enum.Register[ToolType](
	CodeInterpreterToolType,
	RetrievalToolType,
	FunctionToolType,
)

type jsonMappingFunctionTool struct{ Function Function }

func (dto jsonMappingFunctionTool) GetType() ToolType {
	return FunctionToolType
}

func (dto jsonMappingFunctionTool) MarshalJSON() ([]byte, error) {
	type DTO struct {
		Type     ToolType `json:"type" enum:"function"`
		Function Function `json:"function"`
	}
	return json.Marshal(DTO{
		Type:     FunctionToolType,
		Function: dto.Function,
	})
}

type ResponseFormat struct {
	Type string `json:"type"`
}

func JSONResponseFormat() *ResponseFormat {
	return &ResponseFormat{Type: "json_object"}
}

func TextResponseFormat() *ResponseFormat {
	return &ResponseFormat{Type: "text"}
}

///////////////////////////////////////////////////// Tool Choice /////////////////////////////////////////////////////

type ToolChoiceID string

var _ = enum.Register[ToolChoiceID](
	toolChoiceIDNone,
	toolChoiceIDAuto,
	toolChoiceIDFunction,
)

type ToolChoice interface {
	ToolChoiceID() ToolChoiceID
	json.Marshaler
}

type ToolCallID string

type ToolCall struct {
	ID           ToolCallID    `json:"id"`
	Type         ToolType      `json:"type"`
	FunctionCall *FunctionCall `json:"function,omitempty"`
}

var _ = logger.RegisterFieldType[ToolCall](func(call ToolCall) logger.LoggingDetail {
	fields := logger.Fields{
		"id":   string(call.ID),
		"type": string(call.Type),
	}
	if call.Type == FunctionToolType {
		fields["function"] = pointer.Deref(call.FunctionCall)
	}
	return fields
})

func (tc ToolCall) LookupFunctionCall() (FunctionCall, bool) {
	if tc.Type != FunctionToolType || tc.FunctionCall == nil {
		return FunctionCall{}, false
	}
	return *tc.FunctionCall, true
}

// NoneToolChoice means the model will not call a function and instead generates a message.
type NoneToolChoice struct{}

const toolChoiceIDNone ToolChoiceID = "none"

func (NoneToolChoice) ToolChoiceID() ToolChoiceID {
	return toolChoiceIDNone
}

func (NoneToolChoice) MarshalJSON() ([]byte, error) {
	return json.Marshal(toolChoiceIDNone)
}

// AutoToolChoice means the model can pick between generating a message or calling a function.
type AutoToolChoice struct{}

const toolChoiceIDAuto ToolChoiceID = "auto"

func (AutoToolChoice) ToolChoiceID() ToolChoiceID {
	return toolChoiceIDAuto
}

func (AutoToolChoice) MarshalJSON() ([]byte, error) {
	return json.Marshal(toolChoiceIDAuto)
}

// FunctionToolChoice Will tell GPT to use a specific function from the supplied tooling.
type FunctionToolChoice struct {
	// Name of the function that needs to be executed
	Name FunctionName
}

const toolChoiceIDFunction ToolChoiceID = "function"

func (tc FunctionToolChoice) ToolChoiceID() ToolChoiceID {
	return toolChoiceIDFunction
}

func (tc FunctionToolChoice) MarshalJSON() ([]byte, error) {
	type DTOFunction struct {
		Name string `json:"name"`
	}
	type DTO struct {
		Type     string      `json:"type"`
		Function DTOFunction `json:"function"`
	}
	return json.Marshal(DTO{
		Type: string(tc.ToolChoiceID()),
		Function: DTOFunction{
			Name: string(tc.Name),
		},
	})
}

//{"error":{"message":"rate_limit_exceeded: Request too large for gpt-3.5-turbo in organization org-Utp59DiEAZIBtPTEME47y6hP on tokens per min (TPM): Limit 160000, Requested 161170. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more."},"func":"go.llib.dev/llmate.Task.Act","index":0,"level":"debug","message":"If.Check.asking","timestamp":"2023-11-20T13:34:55+01:00"}
//{"error":{"message":"error occured during llmate.Task execution: rate_limit_exceeded: Request too large for gpt-3.5-turbo in organization org-Utp59DiEAZIBtPTEME47y6hP on tokens per min (TPM): Limit 160000, Requested 161170. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more."},"level":"fatal","message":"application error","timestamp":"2023-11-20T13:34:55+01:00"}
