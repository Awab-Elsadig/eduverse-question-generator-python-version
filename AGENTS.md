# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Problem Extractor — a single-service Python/FastAPI web app that extracts, structures, reviews, and exports end-of-chapter problems from textbook PDFs. No database; uses filesystem (`output/` directory). AI provider keys (Groq, Gemini) are optional — the main workflow is manual copy-paste with gemini.google.com.

### Running the dev server

```
python3 main.py
```

Starts uvicorn on `0.0.0.0:8001` with `--reload`. Note: use `python3`, not `python` (no `python` symlink in the default environment).

### Linting

No linting config is included in the repo. Use `ruff check *.py` for quick linting.

### Testing

No automated test suite exists. Test manually via the web UI at `http://localhost:8001` or by hitting API endpoints with `curl`.

Key API endpoints for quick smoke tests:
- `GET /` — returns the HTML SPA
- `GET /api/pdf-info` — returns `{"saved": false, "name": null}` when no PDF is saved
- `POST /api/parse-json` — accepts a JSON array as `text/plain` body, returns parsed problems

### Gotchas

- `main.py` contains a `SyntaxWarning: invalid escape sequence '\s'` in the inline HTML string. This is benign — it comes from the embedded JavaScript regex and does not affect functionality.
- The `tempfile` import in `main.py` is unused (ruff will flag it). This is existing code, not a regression.
- External AI APIs (Groq, Gemini) are optional. The manual workflow (copy prompt to gemini.google.com, paste JSON back) works without any API keys.
