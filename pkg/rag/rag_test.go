package rag

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func loadTestEngine(t *testing.T) *Engine {
	t.Helper()
	kbPath := filepath.Join("..", "..", "kb", "knowledge_base.json")
	eng, err := NewEngine(Config{
		KBFile:        kbPath,
		Mode:          "layered",
		MinSim:        0.5,
		ExampleMaxLen: 600,
	})
	require.NoError(t, err)
	require.Len(t, eng.Entries, 42)
	return eng
}

func TestDecideLayers(t *testing.T) {
	eng := loadTestEngine(t)

	// 1. Random scanner paths -> "404"
	d := eng.Decide("GET", "/aaa9")
	assert.Equal(t, "404", d.Layer)
	assert.Nil(t, d.Entry)

	d = eng.Decide("GET", "/%23")
	assert.Equal(t, "404", d.Layer)

	// /HNAP1 is uppercase and must NOT match RandomPathRx
	d = eng.Decide("GET", "/HNAP1")
	assert.Equal(t, "supervised", d.Layer)
	require.NotNil(t, d.Entry)
	assert.Equal(t, "R41", d.Entry.ID)

	// 2. Static paths -> "cache" or "cache-ico"
	d = eng.Decide("GET", "/robots.txt")
	assert.Equal(t, "cache", d.Layer)
	require.NotNil(t, d.Entry)
	assert.Equal(t, "R19", d.Entry.ID)

	d = eng.Decide("GET", "/sitemap.xml")
	assert.Equal(t, "cache", d.Layer)
	require.NotNil(t, d.Entry)
	assert.Equal(t, "R20", d.Entry.ID)

	d = eng.Decide("GET", "/favicon.ico")
	assert.Equal(t, "cache-ico", d.Layer)
	assert.Nil(t, d.Entry)

	// 3. Exact JSON API match (including aliases and query strings) -> "cache-api"
	d = eng.Decide("GET", "/swagger.json")
	assert.Equal(t, "cache-api", d.Layer)
	require.NotNil(t, d.Entry)
	assert.Equal(t, "R21", d.Entry.ID)

	d = eng.Decide("GET", "/pods")
	assert.Equal(t, "cache-api", d.Layer)
	require.NotNil(t, d.Entry)
	assert.Equal(t, "R38", d.Entry.ID)

	d = eng.Decide("GET", "/webapi/entry.cgi?api=SYNO.API.Info&version=1&method=query")
	assert.Equal(t, "cache-api", d.Layer)
	require.NotNil(t, d.Entry)
	assert.Equal(t, "R42", d.Entry.ID)

	// 4. Method differentiation (GET vs POST /wp-login.php, GET vs POST /mcp)
	dGet := eng.Decide("GET", "/wp-login.php")
	assert.Equal(t, "supervised", dGet.Layer)
	require.NotNil(t, dGet.Entry)
	assert.Equal(t, "R15", dGet.Entry.ID)

	dPost := eng.Decide("POST", "/wp-login.php")
	assert.Equal(t, "supervised", dPost.Layer)
	require.NotNil(t, dPost.Entry)
	assert.Equal(t, "R04", dPost.Entry.ID)

	// 5. Root "/" only matches root "GET /" (R01), not "POST /"
	dRootGet := eng.Decide("GET", "/")
	assert.Equal(t, "supervised", dRootGet.Layer)
	require.NotNil(t, dRootGet.Entry)
	assert.Equal(t, "R01", dRootGet.Entry.ID)

	dRootPost := eng.Decide("POST", "/")
	assert.Equal(t, "free", dRootPost.Layer)
	assert.Nil(t, dRootPost.Entry)

	// 6. Unknown family -> "free"
	dFree := eng.Decide("GET", "/owa/")
	assert.Equal(t, "free", dFree.Layer)
	assert.Nil(t, dFree.Entry)
}

func TestExampleEssence(t *testing.T) {
	eng := loadTestEngine(t)

	// HTML essence (phpinfo R05: 82KB -> <= 600 chars, no base64 images)
	var r05, r03, r11, r38 Entry
	for _, e := range eng.Entries {
		switch e.ID {
		case "R05":
			r05 = e
		case "R03":
			r03 = e
		case "R11":
			r11 = e
		case "R38":
			r38 = e
		}
	}

	essR05 := ExampleEssence(r05.Body, r05.ContentType, 600)
	assert.LessOrEqual(t, len(essR05), 600)
	assert.NotContains(t, essR05, "data:image/png;base64")

	// WordPress login R03: preserves <form>
	essR03 := ExampleEssence(r03.Body, r03.ContentType, 600)
	assert.LessOrEqual(t, len(essR03), 600)
	assert.Contains(t, strings.ToLower(essR03), "<form")

	// SQL dump R11: starts with -- MariaDB dump\nCREATE TABLE
	essR11 := ExampleEssence(r11.Body, r11.ContentType, 600)
	assert.LessOrEqual(t, len(essR11), 600)
	assert.True(t, strings.HasPrefix(essR11, "-- MariaDB dump\nCREATE TABLE"))

	// JSON pods R38: shrunk to <= 600 chars while remaining valid JSON
	essR38 := ExampleEssence(r38.Body, r38.ContentType, 600)
	assert.LessOrEqual(t, len(essR38), 600)
	assert.True(t, json.Valid([]byte(essR38)), "shrunk JSON must remain valid JSON")
}

func TestGuardrailsAndFallbacks(t *testing.T) {
	eng := loadTestEngine(t)
	r02 := eng.Retrieve("GET", "/.env")
	require.NotNil(t, r02)

	// Valid JSON without leak
	rawOK := "```json\n{\"Headers\": {\"Content-Type\": \"text/plain\"}, \"Body\": \"APP_NAME=Acme\"}\n```"
	respBytes, layer, leak := ResolvePostLLM(rawOK, nil, r02, "/.env", "supervised")
	assert.Equal(t, "supervised", layer)
	assert.Empty(t, leak)
	assert.Contains(t, string(respBytes), "APP_NAME=Acme")

	// Leak of internal IP 10.100.100.21 with KB entry -> fallback-ref
	rawLeak := `{"Headers": {"Content-Type": "text/plain"}, "Body": "DB_HOST=10.100.100.21"}`
	respBytes, layer, leak = ResolvePostLLM(rawLeak, nil, r02, "/.env", "supervised")
	assert.Equal(t, "fallback-ref", layer)
	assert.NotEmpty(t, leak)
	assert.Contains(t, string(respBytes), "203.0.113.20")

	// Invalid JSON without KB entry -> fallback-404
	respBytes, layer, _ = ResolvePostLLM("not a json", nil, nil, "/unknown", "free")
	assert.Equal(t, "fallback-404", layer)
	assert.Contains(t, string(respBytes), "404 Not Found")

	// Allowed exception: "/cowrie/" in requestPath allows "cowrie" in response
	rawCowrie := `{"Headers": {"Content-Type": "text/html"}, "Body": "<html>cowrie login</html>"}`
	_, layer, leak = ResolvePostLLM(rawCowrie, nil, nil, "/cowrie/admin", "free")
	assert.Equal(t, "free", layer)
	assert.Empty(t, leak)
}
