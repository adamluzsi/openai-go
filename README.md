# Go OpenAI Kit Package

## Overview

The `openai` package is a Go client library for interacting with OpenAI's GPT models. It provides a convenient way to make chat completion requests to different versions of the GPT model, including GPT-3.5 Turbo and GPT-4.

## Features

- Supports multiple GPT models (GPT-3.5 Turbo, GPT-4, Babbage, etc.)
- Allows for conversation-based interactions using `ChatSession`
- Provides retry strategies for API requests
- Customizable HTTP client
- Environment variable support for API key

## Installation

To install the package, run:

```go
import "go.llib.dev/openai"
```

## Local development

To set up local development:

```sh
export OPENAI_API_KEY="your-openai-api-key-goes-here"
. .envrc # or direnv allow
```

## Usage

### Initialize the Client

```go
client := &openai.Client{
    APIKey: "your-openai-api-key", // or leave it and OPENAI_API_KEY from the env will be used
}
```

### Create a Chat Session

```go
session := openai.ChatSession{Model: openai.GPT3}.
	WithSystemMessage("You are a helpful assistant.")
```

### Make a Chat Completion Request

```go
session, err := client.ChatSession(ctx, session)
if err != nil {
    log.Fatal(err)
}
```

### Access the Assistant's Last Message

```go
lastReply := newSession.LastAssistantContent()
```

## Error Handling

The package defines custom errors like `ErrContextLengthExceeded` for better error handling.
It also uses a generic http error retry strategy, in attempt to handle common networking issues.

## Contributing

Feel free to open issues or submit pull requests.
