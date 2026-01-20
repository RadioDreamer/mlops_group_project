# `Tests`

This page describes the test suite and how to run tests.

## Running tests

From the project root run (use the repository package manager). To install runtime
and developer dependencies and run tests:

```bash
uv sync
uv sync --dev   # (optional) install development dependencies
uv run pytest -q
# or just invoke
uv run invoke test #(more on the invocations documentation page)
```

## Test structure

- Tests live in the `tests/` directory.
- `tests/test_api.py` uses `fastapi.testclient.TestClient` to exercise the API endpoints and patches environment variables.

## Writing tests

- Use `pytest` and fixtures for reusable setup.
- Use `unittest.mock.patch` to stub external dependencies (e.g., GCS, wandb, or the model callable).
- When testing FastAPI endpoints, use `TestClient(app)` and assert status codes and JSON bodies.

## Notes

- Tests may interact with SQLite files; ensure test fixtures clean or isolate test DB paths.
- Keep network calls mocked to avoid flakiness.
