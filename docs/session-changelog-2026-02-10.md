# Session Changelog (2026-02-10)

- Date: 2026-02-10
- Author: Codex
- Scope: CLI / OpenAI-compatible translator / EPUB loader / token estimation / retry & resume workflow

## Functional Changes

### 1. OpenAI-compatible custom model names are now accepted directly

- Before: `--model` only accepted predefined choices and rejected values like `DeepSeek-V3.2`.
- Now: any model name can be passed via `--model <name>` when `--api_base` is provided.
- Behavior:
  - Unknown `--model` + no `--api_base` -> parser error.
  - Unknown `--model` + `--api_base` -> internally routed through OpenAI-compatible translator and uses the given model name.

## 2. Rate-limit controls and gateway cooldown controls were added

- New CLI options:
  - `--rpm` (request-per-minute cap; 0 means disabled)
  - `--gateway-cooldown-threshold`
  - `--gateway-cooldown-seconds`
- OpenAI-compatible translator now:
  - enforces request spacing (`set_interval` / `set_rpm`)
  - retries transient failures with exponential backoff
  - detects repeated 504 errors and applies cooldown

## 3. Batch translation robustness improved

- `translate_list` now uses adaptive recursion:
  - tries batch translation first
  - if parsing or request fails, splits batch into smaller chunks
  - if chunk size reaches 1, falls back to single-paragraph translation
- This prevents a whole chapter from failing due to one bad batch.

## 4. EPUB accumulated translation now supports dynamic backoff/recovery

- New loader-side adaptive controls:
  - `accumulated_min_num`
  - `accumulated_backoff_factor`
  - `accumulated_recover_factor`
  - `accumulated_recover_successes`
- New CLI options:
  - `--accumulated-min-num`
  - `--accumulated-backoff-factor`
  - `--accumulated-recover-factor`
  - `--accumulated-recover-successes`
- Behavior:
  - when gateway-timeout-like signals are observed, current accumulated budget is reduced
  - after enough stable batches, budget gradually recovers

## 5. Resume reliability improved

- Loader now persists progress more aggressively and exits with non-zero status on unexpected failures.
- Partial output and checkpoint are saved on exception for retry/resume.
- CLI auto-enables resume when a checkpoint file already exists beside the source book.

## 6. Token estimation now adapts to actual model names

- Before: token estimator was effectively tied to `gpt-3.5-turbo-0301` and could fail for unknown names.
- Now:
  - estimator resolves model from runtime settings (`--model`, `--model_list`, custom model name, ollama model)
  - unknown model names fall back to `cl100k_base`
  - estimator is model-aware but still heuristic (not billing-grade)

## 7. Gemini import warning reduction

- Gemini translator import was switched to lazy loading in `translator/__init__.py`.
- Result: Gemini package warning is no longer emitted at startup unless Gemini is actually used.

## Test Coverage Added

- `tests/test_cli_custom_model.py`
  - custom model via `--api_base`
  - auto-resume checkpoint detection
  - adaptive accumulated CLI option wiring
- `tests/test_chatgptapi_retry.py`
  - retry behavior
  - cooldown behavior
  - gateway-timeout event consumption
  - adaptive split in batch translation
- `tests/test_epub_resume.py`
  - failure checkpoint + resume continuation
- `tests/test_epub_adaptive_accumulated.py`
  - automatic accumulated backoff under gateway timeout signals
- `tests/test_utils_token_estimator.py`
  - unknown/custom model estimator fallback
  - gpt-5-nano estimator support

## Known Non-blocking Test Failures (pre-existing environment issues)

- `tests/test_integration.py::test_google_translate_epub`
  - Windows terminal encoding (`gbk`) with rich output in translator path.
- `tests/test_integration.py::test_deepl_free_translate_epub`
  - upstream `PyDeepLX` / `httpx` argument incompatibility (`proxies`).

These failures are not introduced by this session’s feature changes.
