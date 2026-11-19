package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

type openAIExtraFieldsKey struct{}

// WithOpenAIExtraFields attaches extra JSON fields (e.g., "repetition_penalty": 1.1 for vLLM)
// to the context so they are merged into the outgoing OpenAI/vLLM /chat/completions request body.
func WithOpenAIExtraFields(ctx context.Context, fields map[string]any) context.Context {
	return context.WithValue(ctx, openAIExtraFieldsKey{}, fields)
}

type extraFieldsHTTPClient struct {
	base *http.Client
}

func (c *extraFieldsHTTPClient) Do(req *http.Request) (*http.Response, error) {
	if req.Body != nil && req.Method == http.MethodPost {
		if extra, ok := req.Context().Value(openAIExtraFieldsKey{}).(map[string]any); ok && len(extra) > 0 {
			bodyBytes, err := io.ReadAll(req.Body)
			_ = req.Body.Close()
			if err == nil {
				var payload map[string]any
				if err := json.Unmarshal(bodyBytes, &payload); err == nil && payload != nil {
					for k, v := range extra {
						payload[k] = v
					}
					if updated, err := json.Marshal(payload); err == nil {
						bodyBytes = updated
					}
				}
				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.ContentLength = int64(len(bodyBytes))
				req.Header.Set("Content-Length", strconv.Itoa(len(bodyBytes)))
			}
		}
	}
	return c.base.Do(req)
}

func initOpenAIClient(config Config) (llms.Model, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	opts := []openai.Option{
		openai.WithModel(config.Model),
		openai.WithToken(config.APIKey),
		openai.WithBaseURL(config.ServerURL),
		openai.WithHTTPClient(&extraFieldsHTTPClient{base: http.DefaultClient}),
	}
	m, err := openai.New(opts...)
	if err != nil {
		return nil, err
	}
	return m, nil
}
