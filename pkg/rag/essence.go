package rag

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

var (
	reStripList = []*regexp.Regexp{
		regexp.MustCompile(`(?is)<script\b.*?</script>`),
		regexp.MustCompile(`(?is)<style\b.*?</style>`),
		regexp.MustCompile(`(?is)<head\b.*?</head>`),
		regexp.MustCompile(`(?is)<svg\b.*?</svg>`),
		regexp.MustCompile(`(?is)<img\b[^>]*>`),
		regexp.MustCompile(`(?i)data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+`),
		regexp.MustCompile(`(?s)<!--.*?-->`),
	}

	keepAttrs = map[string]bool{
		"name":        true,
		"type":        true,
		"action":      true,
		"method":      true,
		"value":       true,
		"placeholder": true,
		"for":         true,
	}

	reTag  = regexp.MustCompile(`<(/?)([a-zA-Z][\w:-]*)([^>]*)>`)
	reAttr = regexp.MustCompile(`([\w:-]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s"'>]+))?`)

	reComments = []*regexp.Regexp{
		regexp.MustCompile(`(?s)/\*.*?\*/`),
		regexp.MustCompile(`(?m)^\s*(//|#)[^\n]*\n?`),
	}

	reBodyBlock   = regexp.MustCompile(`(?is)<body\b.*?</body>`)
	reFormBlock   = regexp.MustCompile(`(?is)<form\b.*?</form>`)
	reSpacesTabs  = regexp.MustCompile(`[ \t]+`)
	reBlankLines  = regexp.MustCompile(`\n\s*\n+`)
)

// slimTags removes decorative HTML attributes (class, id, style, aria-*, data-*),
// keeping only attributes that define form semantics.
func slimTags(html string) string {
	return reTag.ReplaceAllStringFunc(html, func(match string) string {
		sub := reTag.FindStringSubmatch(match)
		if len(sub) != 4 {
			return match
		}
		slash, tag, attrs := sub[1], sub[2], sub[3]
		var kept []string
		for _, a := range reAttr.FindAllStringSubmatch(attrs, -1) {
			k := strings.ToLower(a[1])
			if keepAttrs[k] {
				kept = append(kept, a[0])
			}
		}
		if len(kept) > 0 {
			return fmt.Sprintf("<%s%s %s>", slash, tag, strings.Join(kept, " "))
		}
		return fmt.Sprintf("<%s%s>", slash, tag)
	})
}

func compactJSONString(v any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return ""
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

func shrinkValue(v any, budget int) any {
	if budget < 30 {
		budget = 30
	}
	switch val := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make(map[string]any)
		for _, k := range keys {
			cand := make(map[string]any, len(out)+1)
			for ck, cv := range out {
				cand[ck] = cv
			}
			subBudget := budget / 2
			if subBudget < 30 {
				subBudget = 30
			}
			cand[k] = shrinkValue(val[k], subBudget)
			if len(compactJSONString(cand)) > budget && len(out) > 0 {
				break
			}
			out = cand
		}
		return out
	case []any:
		var out []any
		for _, sub := range val {
			subBudget := budget / 2
			if subBudget < 30 {
				subBudget = 30
			}
			cand := append(append([]any(nil), out...), shrinkValue(sub, subBudget))
			if len(compactJSONString(cand)) > budget && len(out) > 0 {
				break
			}
			out = cand
		}
		return out
	case string:
		if len(val) > 80 {
			return val[:77] + "..."
		}
		return val
	default:
		return v
	}
}

func shrinkJSON(raw string, maxLen int) (string, error) {
	var obj any
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return "", err
	}
	dumped := compactJSONString(obj)
	if len(dumped) <= maxLen {
		return dumped, nil
	}

	switch val := obj.(type) {
	case map[string]any:
		n := len(val)
		if n < 1 {
			n = 1
		}
		perKeyBudget := maxLen / n
		if perKeyBudget < 40 {
			perKeyBudget = 40
		}
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		shrunk := make(map[string]any, len(val))
		for _, k := range keys {
			shrunk[k] = shrinkValue(val[k], perKeyBudget)
		}
		for len(keys) > 0 && len(compactJSONString(shrunk)) > maxLen {
			last := keys[len(keys)-1]
			keys = keys[:len(keys)-1]
			delete(shrunk, last)
		}
		return compactJSONString(shrunk), nil
	case []any:
		return compactJSONString(shrinkValue(val, maxLen)), nil
	default:
		if len(dumped) > maxLen {
			return dumped[:maxLen], nil
		}
		return dumped, nil
	}
}

// ExampleEssence extracts the semantic essence of a KB entry body within maxLen chars,
// mirroring example_essence() from honeypot-llm-bench.
func ExampleEssence(body, contentType string, maxLen int) string {
	if maxLen <= 0 {
		maxLen = DefaultExampleMaxLen
	}
	text := body
	ct := strings.ToLower(contentType)

	checkLen := len(text)
	if checkLen > 6000 {
		checkLen = 6000
	}

	// 1. SQL dumps: start at first CREATE TABLE
	if strings.Contains(ct, "sql") || strings.Contains(text[:checkLen], "CREATE TABLE") {
		if idx := strings.Index(text, "CREATE TABLE"); idx > 0 {
			text = "-- MariaDB dump\n" + text[idx:]
		}
		return truncateString(text, maxLen)
	}

	// 2. JSON payloads: shrink keeping valid JSON
	trimmedLeft := strings.TrimLeft(text, " \t\r\n")
	if strings.Contains(ct, "json") || strings.HasPrefix(trimmedLeft, "{") || strings.HasPrefix(trimmedLeft, "[") {
		if shrunk, err := shrinkJSON(text, maxLen); err == nil {
			return shrunk
		}
		return truncateString(text, maxLen)
	}

	// 3. Plain text / PHP config files: strip comment blocks
	if strings.Contains(ct, "text/plain") || strings.HasPrefix(trimmedLeft, "<?php") {
		stripped := text
		for _, rx := range reComments {
			stripped = rx.ReplaceAllString(stripped, "")
		}
		stripped = strings.TrimSpace(reBlankLines.ReplaceAllString(stripped, "\n"))
		if len(stripped) >= 40 {
			text = stripped
		}
		return truncateString(text, maxLen)
	}

	// 4. HTML pages: strip scripts/styles/head/svg/img/base64/comments, slim attributes, prioritize <form>
	headCheck := len(text)
	if headCheck > 200 {
		headCheck = 200
	}
	if strings.Contains(ct, "html") || strings.Contains(strings.ToLower(text[:headCheck]), "<html") {
		for _, rx := range reStripList {
			text = rx.ReplaceAllString(text, "")
		}
		text = slimTags(text)
		if m := reBodyBlock.FindString(text); m != "" {
			text = m
		}
		text = reSpacesTabs.ReplaceAllString(text, " ")
		text = strings.TrimSpace(reBlankLines.ReplaceAllString(text, "\n"))

		if loc := reFormBlock.FindStringIndex(text); loc != nil {
			form := text[loc[0]:loc[1]]
			if loc[1] > maxLen {
				prefEnd := loc[0]
				if prefEnd > 150 {
					prefEnd = 150
				}
				if prefEnd > 0 {
					prefix := text[:prefEnd]
					text = prefix + "\n" + form
				} else {
					text = form
				}
			}
		}
	}

	return truncateString(text, maxLen)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// BuildUserPrompt constructs the user prompt for the LLM following honeypot-llm-bench's build_user_prompt_ex
// and appends the Galah anti-prompt-injection tail.
func BuildUserPrompt(httpRequest, ragMode string, entry *Entry, maxLen int) (string, bool) {
	base := fmt.Sprintf("Analyze the following HTTP request and generate the response:\n\n%s", strings.TrimSpace(httpRequest))
	if ragMode == "none" || ragMode == "free" || entry == nil {
		return base + GalahTail, false
	}

	ctype := entry.ContentType
	example := ExampleEssence(entry.Body, ctype, maxLen)

	if ragMode == "supervised" || ragMode == "layered" {
		prompt := fmt.Sprintf(
			"%s\n\n"+
				"Below is the ACTUAL body a real server returned for a similar request "+
				"(Content-Type: %s). Use it as the value of the \"Body\" field of "+
				"your JSON answer, as a single string (escape quotes and newlines). Keep "+
				"the outer format {\"Headers\": {...}, \"Body\": \"...\"} exactly; never "+
				"return the reference as the top-level object. Reproduce the content, "+
				"changing ONLY sensitive values: real IPs become 192.0.2.x / 198.51.100.x "+
				"/ 203.0.113.x, real hostnames become app01.internal.example.com, "+
				"credentials stay fictional. Add no notes.\n\n"+
				"--- REFERENCE BODY START ---\n%s\n--- REFERENCE BODY END ---%s",
			base, ctype, example, GalahTail,
		)
		return prompt, true
	}

	// "full" mode: reference as style/structure guide
	prompt := fmt.Sprintf(
		"%s\n\n"+
			"For reference, a real server responded to a similar request as follows "+
			"(Content-Type: %s). Use it as a style/structure guide, but generate "+
			"your own response with fictional values:\n\n"+
			"--- REFERENCE START ---\n%s\n--- REFERENCE END ---%s",
		base, ctype, example, GalahTail,
	)
	return prompt, true
}
