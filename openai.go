package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/adamluzsi/frameless/pkg/errorkit"
	"github.com/adamluzsi/frameless/pkg/httpkit"
	"github.com/adamluzsi/frameless/pkg/retry"
	"io"
	"net/http"
	"os"
	"time"
)

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
	BaseURL string
	APIKey  string

	HTTPClient    *http.Client
	RetryStrategy retry.Strategy
}

var DefaultRetryStrategy retry.Strategy = retry.Jitter{
	MaxRetries:      9,
	MaxWaitDuration: 3 * time.Second,
}

func (c *Client) getRetryStrategy() retry.Strategy {
	if c.RetryStrategy == nil {
		return DefaultRetryStrategy
	}
	return c.RetryStrategy
}

// getBaseURL returns the base URL for API requests.
func (c *Client) getBaseURL() string {
	if len(c.BaseURL) == 0 {
		c.BaseURL = defaultBaseURL
	}
	return c.BaseURL
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
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{
		Transport: httpkit.RetryRoundTripper{
			Transport:     http.DefaultTransport,
			RetryStrategy: c.getRetryStrategy(),
		},
		Timeout: 30 * time.Second,
	}
}

type ChatSession struct {
	// Model specifies the ID of the model to use (e.g., "text-davinci-002").
	Model ChatModelID `json:"model"`

	// Messages is an array of ChatMessage structs representing the conversation history.
	Messages []ChatMessage `json:"messages"`
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
}

type ChatMessageRole string

const (
	SystemChatMessageRole    ChatMessageRole = "system"
	UserChatMessageRole      ChatMessageRole = "user"
	AssistantChatMessageRole ChatMessageRole = "assistant"
)

// ChatMessage represents a single message in the conversation history.
type ChatMessage struct {
	// Role specifies the role of the message sender, usually "system", "user", or "assistant".
	Role ChatMessageRole `json:"role" enum:"system;user;assistant;"`

	// Content contains the actual text of the message.
	Content string `json:"content"`
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
	FinishReason string `json:"finish_reason"`

	// Index is the index of the choice in the array.
	// Example: 0
	Index int `json:"index"`
}

func (c *Client) ChatSession(ctx context.Context, session ChatSession) (ChatSession, error) {
	// Create a deep copy of the original ChatSession's Messages to avoid modifying it
	session = session.Clone()

call:
	response, err := c.ChatCompletion(ctx, ChatCompletionRequest{
		Model:    session.Model,
		Messages: session.Messages,
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

	return session, nil
}

func (c *Client) ChatCompletion(ctx context.Context, cc ChatCompletionRequest) (ChatCompletionResponse, error) {
	var reply ChatCompletionResponse

	jsonPayload, err := json.Marshal(cc)
	if err != nil {
		return reply, err
	}

	uri := c.getBaseURL() + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", uri, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return reply, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.getAPIKey())

	client := c.getHTTPClient()

	resp, err := client.Do(req)
	if err != nil {
		return reply, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return reply, err
	}

	if resp.StatusCode != http.StatusOK {
		var errResp errorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return reply, err
		}
		if errResp.Error.Code == errorCodeContextLengthExceeded {
			return reply, fmt.Errorf("%w: %s",
				ErrContextLengthExceeded, errResp.Error.Message)
		}
		return reply, fmt.Errorf(errResp.Error.Message)
	}

	if err := json.Unmarshal(body, &reply); err != nil {
		return reply, err
	}

	return reply, nil
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
