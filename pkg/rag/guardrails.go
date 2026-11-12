package rag

import (
	"encoding/json"
	"regexp"
	"strings"
)

// ResponsePayload represents the normalized JSON response structure used by Galah.
type ResponsePayload struct {
	Headers map[string]string `json:"Headers"`
	Body    string            `json:"Body"`
	Status  int               `json:"Status,omitempty"`
}

type forbiddenPattern struct {
	raw string
	rx  *regexp.Regexp
}

var (
	reCodeFence = regexp.MustCompile("(?i)```(?:json)?")

	forbiddenPatterns = []forbiddenPattern{
		{raw: `\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`, rx: regexp.MustCompile(`(?i)\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`)},
		{raw: `\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b`, rx: regexp.MustCompile(`(?i)\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b`)},
		{raw: `\b192\.168\.\d{1,3}\.\d{1,3}\b`, rx: regexp.MustCompile(`(?i)\b192\.168\.\d{1,3}\.\d{1,3}\b`)},
		{raw: `\bdpipot\b`, rx: regexp.MustCompile(`(?i)\bdpipot\b`)},
		{raw: `\bcowrie\b`, rx: regexp.MustCompile(`(?i)\bcowrie\b`)},
		{raw: `\bwordpot\b`, rx: regexp.MustCompile(`(?i)\bwordpot\b`)},
		{raw: `\bheralding\b`, rx: regexp.MustCompile(`(?i)\bheralding\b`)},
		{raw: `\.svc\.cluster\.local`, rx: regexp.MustCompile(`(?i)\.svc\.cluster\.local`)},
		{raw: `\bkafka-svc\b`, rx: regexp.MustCompile(`(?i)\bkafka-svc\b`)},
	}
)

// ExtractJSON extracts and normalizes a JSON response object from raw LLM output,
// mirroring extract_json() in benchmark_galah_ollama.py.
func ExtractJSON(text string) (*ResponsePayload, bool) {
	cleaned := strings.TrimSpace(reCodeFence.ReplaceAllString(text, ""))
	if cleaned == "" {
		return nil, false
	}

	if p, ok := parseResponseMap(cleaned); ok {
		return p, true
	}

	start := strings.IndexByte(cleaned, '{')
	if start == -1 {
		return nil, false
	}

	depth := 0
	inStr := false
	esc := false
	for i := start; i < len(cleaned); i++ {
		c := cleaned[i]
		if inStr {
			if esc {
				esc = false
			} else if c == '\\' {
				esc = true
			} else if c == '"' {
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
		} else if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				candidate := cleaned[start : i+1]
				if p, ok := parseResponseMap(candidate); ok {
					return p, true
				}
				return nil, false
			}
		}
	}
	return nil, false
}

func parseResponseMap(candidate string) (*ResponsePayload, bool) {
	var raw map[string]any
	if err := json.Unmarshal([]byte(candidate), &raw); err != nil || raw == nil {
		return nil, false
	}

	var headersRaw any
	var bodyRaw any
	var statusRaw any

	for k, v := range raw {
		switch strings.ToLower(k) {
		case "headers":
			headersRaw = v
		case "body":
			bodyRaw = v
		case "status":
			statusRaw = v
		}
	}

	headersMap, ok := headersRaw.(map[string]any)
	if !ok || bodyRaw == nil {
		return nil, false
	}

	headers := make(map[string]string, len(headersMap))
	for k, v := range headersMap {
		if s, ok := v.(string); ok {
			headers[k] = s
		} else {
			b, _ := json.Marshal(v)
			headers[k] = string(b)
		}
	}
	if len(headers) == 0 {
		return nil, false
	}

	var bodyStr string
	switch b := bodyRaw.(type) {
	case string:
		bodyStr = b
	default:
		marshaled, err := json.Marshal(b)
		if err != nil {
			return nil, false
		}
		bodyStr = string(marshaled)
	}

	if bodyStr == "" {
		return nil, false
	}

	status := 0
	if st, ok := statusRaw.(float64); ok {
		status = int(st)
	}

	return &ResponsePayload{
		Headers: headers,
		Body:    bodyStr,
		Status:  status,
	}, true
}

// ForbiddenHit checks if serialized output contains any forbidden internal infrastructure pattern,
// unless the matched substring literally appears in the attacker's requestPath.
func ForbiddenHit(serialized, requestPath string) string {
	reqLower := strings.ToLower(requestPath)
	for _, pat := range forbiddenPatterns {
		m := pat.rx.FindString(serialized)
		if m != "" && !strings.Contains(reqLower, strings.ToLower(m)) {
			return pat.raw
		}
	}
	return ""
}

// NotFoundResponse returns the serialized JSONResponse for an Apache 404 Not Found page.
func NotFoundResponse() ([]byte, error) {
	payload := ResponsePayload{
		Headers: map[string]string{
			"Content-Type": "text/html; charset=iso-8859-1",
			"Server":       "Apache/2.4.41 (Ubuntu)",
		},
		Body:   NotFoundBody,
		Status: 404,
	}
	return marshalNoEscape(payload)
}

// ReferenceResponse returns the serialized JSONResponse built directly from a KB entry.
func ReferenceResponse(entry *Entry) ([]byte, error) {
	ct := "text/plain"
	if entry != nil && entry.ContentType != "" {
		ct = entry.ContentType
	}
	body := ""
	if entry != nil {
		body = entry.Body
	}
	payload := ResponsePayload{
		Headers: map[string]string{
			"Content-Type": ct,
			"Server":       "Apache/2.4.58 (Debian)",
		},
		Body: body,
	}
	return marshalNoEscape(payload)
}

// ResolvePostLLM validates the LLM output for valid JSON and infrastructure leakage,
// applying the exact fallback rules from galah_proxy.py:
//   - Invalid JSON or leak when KB entry exists -> "fallback-ref" (returns KB entry)
//   - Invalid JSON when no KB entry exists      -> "fallback-404" (returns Apache 404)
//   - Leak when no KB entry exists              -> "fallback-leak" (replaces Body with Apache 404)
func ResolvePostLLM(rawLLMOutput string, llmErr error, entry *Entry, requestPath, initialLayer string) ([]byte, string, string) {
	var parsed *ResponsePayload
	var ok bool
	var leak string

	if llmErr == nil && rawLLMOutput != "" {
		parsed, ok = ExtractJSON(rawLLMOutput)
		leak = ForbiddenHit(rawLLMOutput, requestPath)
	}

	if (!ok || leak != "") && entry != nil {
		b, _ := ReferenceResponse(entry)
		return b, "fallback-ref", leak
	}
	if !ok && entry == nil {
		b, _ := NotFoundResponse()
		return b, "fallback-404", leak
	}
	if leak != "" && parsed != nil {
		hdrs := parsed.Headers
		if len(hdrs) == 0 {
			hdrs = map[string]string{"Content-Type": "text/html"}
		}
		b, _ := marshalNoEscape(ResponsePayload{
			Headers: hdrs,
			Body:    NotFoundBody,
		})
		return b, "fallback-leak", leak
	}

	b, _ := marshalNoEscape(*parsed)
	return b, initialLayer, ""
}

func marshalNoEscape(v any) ([]byte, error) {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return []byte(strings.TrimSuffix(buf.String(), "\n")), nil
}
