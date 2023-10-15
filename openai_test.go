package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/adamluzsi/testcase"
	"github.com/adamluzsi/testcase/clock/timecop"
	"github.com/adamluzsi/testcase/let"
	"github.com/adamluzsi/testcase/random"
	"go.llib.dev/openai"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/adamluzsi/testcase/assert"
)

const (
	CheapestChatModel = openai.GPT3
	exampleToken      = "chimichanga"
)

func TestClient_ChatCompletion_smoke(t *testing.T) {
	client := &openai.Client{}

	// Prepare a simple chat completion request
	req := openai.ChatCompletionRequest{
		Model: CheapestChatModel,
		Messages: []openai.ChatMessage{
			{
				Role:    "system",
				Content: "This is an integration test, try to follow the user instructions closely.",
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("please, reply back to me only with '%s'", exampleToken),
			},
		},
	}

	// Make the API call
	resp, err := client.ChatCompletion(context.Background(), req)
	assert.NoError(t, err)
	assert.Contain(t, resp.Model, req.Model)
	assert.NotEmpty(t, resp.ID)
	assert.Equal(t, "chat.completion", resp.Object)
	assert.NotEmpty(t, resp.Created)
	assert.NotEmpty(t, resp.Usage)
	assert.NotEmpty(t, resp.Usage.CompletionTokens)
	assert.NotEmpty(t, resp.Usage.TotalTokens)
	assert.NotEmpty(t, resp.Usage.PromptTokens)
	assert.NotEmpty(t, resp.Choices)
	assert.OneOf(t, resp.Choices, func(it assert.It, got openai.Choice) {
		assert.Equal(it, "assistant", got.Message.Role)
		assert.Contain(it, strings.ToLower(got.Message.Content), exampleToken)
	})
}

func TestChatSession_with(t *testing.T) {
	session := openai.ChatSession{Model: CheapestChatModel}
	original := session

	assert.Equal(t, 0, len(session.Messages))
	session = session.WithSystemMessage("foo")
	assert.Equal(t, 1, len(session.Messages))
	assert.Empty(t, session.LastAssistantContent())
	session = session.WithUserMessage("bar")
	assert.Equal(t, 2, len(session.Messages))
	assert.Empty(t, session.LastAssistantContent())
	session = session.WithAssistantMessage("baz")
	assert.Equal(t, 3, len(session.Messages))
	assert.Equal(t, "baz", session.LastAssistantContent())
	session = session.WithUserMessage("qux")
	assert.Equal(t, 4, len(session.Messages))
	assert.Equal(t, "baz", session.LastAssistantContent())

	assert.Empty(t, original.Messages, "with methods should not modify the receiver")
	assert.Equal(t, openai.ChatMessage{
		Role:    "system",
		Content: "foo",
	}, session.Messages[0])
	assert.Equal(t, openai.ChatMessage{
		Role:    "user",
		Content: "bar",
	}, session.Messages[1])
	assert.Equal(t, openai.ChatMessage{
		Role:    "assistant",
		Content: "baz",
	}, session.Messages[2])
}

func TestClient_ChatSession_smoke(t *testing.T) {
	client := &openai.Client{}

	session := openai.ChatSession{Model: CheapestChatModel}.
		WithSystemMessage("You are a helpful assistant.").
		WithUserMessage(fmt.Sprintf("please, reply back to me with '%s'", exampleToken)).
		WithAssistantMessage(exampleToken).
		WithUserMessage("thank you, now please reply back only with 'OK'.")
	assert.Equal(t, 4, len(session.Messages))

	session, err := client.ChatSession(context.Background(), session)
	assert.NoError(t, err)
	assert.Equal(t, 4+1, len(session.Messages))

	assert.Must(t).AnyOf(func(a *assert.AnyOf) {
		a.Test(func(it assert.It) { it.Must.Equal(session.LastAssistantContent(), "OK") })
		a.Test(func(it assert.It) { it.Must.Contain(session.LastAssistantContent(), "OK") })
	})
}

func TestClient_ChatCompletion_errorRetry(t *testing.T) {
	timecop.SetSpeed(t, math.MaxFloat64)

	var callCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		defer r.Body.Close()
		var req openai.ChatCompletionRequest
		assert.Should(t).NoError(json.NewDecoder(r.Body).Decode(&req))
		assert.Should(t).NotEmpty(req)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	client := &openai.Client{BaseURL: srv.URL}

	// Prepare a simple chat completion request
	req := openai.ChatCompletionRequest{
		Model: CheapestChatModel,
		Messages: []openai.ChatMessage{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: "Hello, how are you doing?",
			},
		},
	}

	_, err := client.ChatCompletion(context.Background(), req)
	assert.Error(t, err)
	assert.True(t, 1 < atomic.LoadInt32(&callCount),
		"expected that the request will be retried",
		assert.Message(fmt.Sprintf("call count: %v", atomic.LoadInt32(&callCount))))
}

var ExampleClient = testcase.Var[*openai.Client]{
	ID: "openai.Client",
	Init: func(t *testcase.T) *openai.Client {
		return &openai.Client{}
	},
}

func TestClient_ChatSession_automaticModelUpgradeOnTooLongToken(t *testing.T) {
	s := testcase.NewSpec(t)

	var (
		Context      = let.Context(s)
		client       = ExampleClient.Bind(s)
		initialModel = testcase.Let[openai.ChatModelID](s, nil)
		tokens       = testcase.Let[[]string](s, nil)
		session      = testcase.Let(s, func(t *testcase.T) openai.ChatSession {
			return openai.ChatSession{Model: initialModel.Get(t)}.
				WithUserMessage(strings.Join(tokens.Get(t), " "))
		})
	)
	act := func(t *testcase.T) (openai.ChatSession, error) {
		return client.Get(t).ChatSession(Context.Get(t), session.Get(t))
	}

	thenTheModelIsUpgraded := func(s *testcase.Spec) {
		s.Then("the model is upgraded and a new request is made", func(t *testcase.T) {
			session, err := act(t)
			assert.NoError(t, err)
			assert.NotEmpty(t, session.LastAssistantContent())
			assert.NotEqual(t, session.Model, initialModel.Get(t))
		})
	}

	s.Context("GPT3.5 is upgraded to GPT3.5 16k", func(s *testcase.Spec) {
		initialModel.LetValue(s, openai.GPT3)

		tokens.Let(s, func(t *testcase.T) []string {
			var tokens []string
			// 5000 Token is over the limit of GPT3.5-turbo,
			// which is being used here with GPT3
			for i := 0; i < 5000; i++ {
				tokens = append(tokens, t.Random.StringNC(1, random.CharsetAlpha()))
			}
			return tokens
		})

		thenTheModelIsUpgraded(s)
	})

	s.Context("GPT4 (8k) is upgraded to GPT4 32k", func(s *testcase.Spec) {
		s.Before(func(t *testcase.T) {
			SkipNonCheap(t)
			SkipTestForGPT4_32K(t)
		})

		initialModel.LetValue(s, openai.GPT4)

		tokens.Let(s, func(t *testcase.T) []string {
			var tokens []string
			// 9000 Token is over the limit of GPT4,
			// which is being used here with GPT3
			for i := 0; i < 9000; i++ {
				tokens = append(tokens, t.Random.StringNC(1, random.CharsetAlpha()))
			}
			return tokens
		})

		thenTheModelIsUpgraded(s)
	})
}

func SkipTestForGPT4_32K(tb testing.TB) {
	tb.Log("Skipping this test with GPT4-32k as I lack access to the model")
	tb.Log("https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4")
	testcase.SkipUntil(tb, 2023, 12, 30, 12)
}

func SkipNonCheap(tb testing.TB) {
	raw, ok := os.LookupEnv("CHEAP")
	if !ok {
		return
	}
	isCheap, err := strconv.ParseBool(raw)
	if err != nil {
		tb.Skip("skipping test due to the presence of CHEAP env var flag")
	}
	if isCheap {
		tb.Skip("SKIPPING test due to CHEAP flag")
	}
}

func TestClient_ChatSession_contextWithError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := &openai.Client{}
	_, err := client.ChatSession(ctx, openai.ChatSession{Model: CheapestChatModel}.
		WithSystemMessage("You are a helpful assistant."))

	assert.ErrorIs(t, err, ctx.Err())
}

func TestClient_ChatCompletion_contextWithError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := &openai.Client{}
	_, err := client.ChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: CheapestChatModel,
		Messages: []openai.ChatMessage{
			{
				Role:    openai.SystemChatMessageRole,
				Content: "You are awesome.",
			},
		},
	})
	assert.ErrorIs(t, err, ctx.Err())
}

func TestChatFunctionCall_json(t *testing.T) {
	v := openai.ChatFunctionCall{
		Name:      "foo-bar-baz",
		Arguments: `{"hello":"world!"}`,
	}

	data, err := json.Marshal(v)
	assert.NoError(t, err)
	assert.NotEmpty(t, data)

	var got openai.ChatFunctionCall
	assert.NoError(t, json.Unmarshal(data, &got))
	assert.Equal(t, v, got)
}

func TestClient_ChatCompletion_functions(t *testing.T) {
	client := openai.Client{}

	type WeatherRequestDTO struct {
		Country string `json:"country"`
		City    string `json:"city"`
	}

	const funcName = "current-weather"
	functions := []openai.ChatFunction{
		{
			Name:        funcName,
			Description: "Retrieve the current weather.",
			Parameters: openai.JSONSchema{
				Type: "object",
				Properties: map[string]openai.JSONSchemaProperty{
					"country": {
						Type:        "string",
						Description: "The city's name where the weather should be checked",
						Required:    true,
					},
					"city": {
						Type:        "string",
						Description: "the city's name where the weather should be checked",
					},
				},
				Required: []string{"city"},
			},
			Exec: func(ctx context.Context, payload json.RawMessage) (any, error) {
				var dto WeatherRequestDTO
				if err := json.Unmarshal(payload, &dto); err != nil {
					return nil, err
				}
				return map[string]any{"weather": "sunny"}, nil
			},
		},
	}

	req := openai.ChatCompletionRequest{
		Model: CheapestChatModel,
		Messages: []openai.ChatMessage{
			{Role: openai.SystemChatMessageRole, Content: "You are a helpful assistant."},
			{Role: openai.UserChatMessageRole, Content: "How's the current weather in Zürich?"},
		},
		Functions: functions,
	}

	ctx := context.Background()

	resp, err := client.ChatCompletion(ctx, req)
	assert.NoError(t, err)

	var (
		fnCall openai.ChatFunctionCall
		reply  openai.ChatMessage
	)
	assert.OneOf(t, resp.Choices, func(it assert.It, got openai.Choice) {
		it.Must.Equal(got.Message.Role, openai.AssistantChatMessageRole)
		it.Must.NotNil(got.Message.FunctionCall)
		assert.Equal(t, got.Message.FunctionCall.Name, funcName)
		reply = got.Message
		fnCall = *got.Message.FunctionCall
	})

	req.Messages = append(req.Messages, reply)

	args := make(map[string]any)
	assert.NoError(t, json.Unmarshal([]byte(fnCall.Arguments), &args))
	var keys []string
	for key, val := range args {
		assert.NotEmpty(t, val)
		keys = append(keys, key)
	}
	assert.ContainExactly(t, keys, []string{"country", "city"})

	req.Messages = append(req.Messages, openai.ChatMessage{
		Role:         openai.FunctionChatMessageRole,
		FunctionName: funcName,
		Content:      `{"weather":"sunny"}`,
	})

	resp, err = client.ChatCompletion(ctx, req)
	assert.NoError(t, err)

	assert.OneOf(t, resp.Choices, func(it assert.It, got openai.Choice) {
		it.Must.Equal(got.Message.Role, openai.AssistantChatMessageRole)
		it.Must.Contain(got.Message.Content, "sunny")
		it.Must.Contain(got.Message.Content, "Zürich")
	})
}

func TestChatSession_functions(t *testing.T) {
	client := openai.Client{}

	type WeatherRequestDTO struct {
		Country     string   `json:"country"`
		City        string   `json:"city"`
		Temperature []string `json:"temperature"`
	}

	const funcName = "current-weather"
	functions := []openai.ChatFunction{
		{
			Name:        funcName,
			Description: "Retrieve the current weather.",
			Parameters: openai.JSONSchema{
				Type: "object",
				Properties: map[string]openai.JSONSchemaProperty{
					"country": {
						Type:        "string",
						Description: "The city's name where the weather should be checked",
						Required:    true,
					},
					"city": {
						Type:        "string",
						Description: "the city's name where the weather should be checked",
					},
					"temperature": {
						Type:        "array",
						Description: "defines what temperate units should be used",
						Items: &openai.JSONSchemaItems{
							Type: "string",
							Enum: []string{"celsius", "fahrenheit"},
						},
						Required: true,
					},
				},
				Required: []string{"city"},
			},
			Exec: func(ctx context.Context, payload json.RawMessage) (any, error) {
				var dto WeatherRequestDTO
				if err := json.Unmarshal(payload, &dto); err != nil {
					return nil, err
				}
				assert.Should(t).NotEmpty(dto)
				return map[string]any{
					"weather": "sunny",
					"temperature": map[string]any{
						"value": 42,
						"metric": func() string {
							for _, metric := range dto.Temperature {
								return metric
							}
							return "celsius"
						}(),
					},
				}, nil
			},
		},
	}

	session := openai.ChatSession{
		Model: CheapestChatModel,
		Messages: []openai.ChatMessage{
			{Role: openai.SystemChatMessageRole, Content: "You are a helpful assistant."},
			{Role: openai.UserChatMessageRole, Content: "How's the current weather in Zürich?"},
		},
		Functions: functions,
	}

	ctx := context.Background()

	session, err := client.ChatSession(ctx, session)
	assert.NoError(t, err)

	assert.OneOf(t, session.Messages, func(it assert.It, got openai.ChatMessage) {
		it.Must.Equal(got.Role, openai.AssistantChatMessageRole)
		it.Must.NotNil(got.FunctionCall)
		assert.Equal(t, got.FunctionCall.Name, funcName)

		args := make(map[string]any)
		it.Must.NoError(json.Unmarshal([]byte(got.FunctionCall.Arguments), &args))
		var keys []string
		for key, val := range args {
			assert.NotEmpty(t, val)
			keys = append(keys, key)
		}
		assert.ContainExactly(t, keys, []string{"country", "city", "temperature"})
	})

	assert.OneOf(t, session.Messages, func(it assert.It, got openai.ChatMessage) {
		it.Must.Equal(got.Role, openai.AssistantChatMessageRole)
		it.Must.Contain(got.Content, "sunny")
		it.Must.Contain(got.Content, "Zürich")
	})
}

func TestChatSession_functions_ExecRequired(t *testing.T) {
	client := openai.Client{BaseURL: "https://go.llib.dev"}

	type WeatherRequestDTO struct {
		Country string `json:"country"`
		City    string `json:"city"`
	}

	const funcName = "current-weather"
	functions := []openai.ChatFunction{
		{
			Name:        funcName,
			Description: "Retrieve the current weather.",
			Parameters: openai.JSONSchema{
				Type: "object",
				Properties: map[string]openai.JSONSchemaProperty{
					"country": {
						Type:        "string",
						Description: "The city's name where the weather should be checked",
						Required:    true,
					},
					"city": {
						Type:        "string",
						Description: "the city's name where the weather should be checked",
					},
				},
				Required: []string{"city"},
			},
		},
	}

	session := openai.ChatSession{
		Model: CheapestChatModel,
		Messages: []openai.ChatMessage{
			{Role: openai.SystemChatMessageRole, Content: "You are a helpful assistant."},
			{Role: openai.UserChatMessageRole, Content: "How's the current weather in Zürich?"},
		},
		Functions: functions,
	}

	ctx := context.Background()

	_, err := client.ChatSession(ctx, session)
	assert.ErrorIs(t, err, openai.ErrFunctionMissingExec)
}
