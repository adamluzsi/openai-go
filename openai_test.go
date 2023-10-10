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
