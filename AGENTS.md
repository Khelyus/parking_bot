# Repository Guidelines

## Project Structure & Module Organization
`parking_bot/` contains the application package: bot startup in `main.py`, Telegram handlers in `telegram_bot.py`, monitoring logic in `service.py`, detection in `detector.py`, and persistence in `repository.py`. Configuration lives in `config/`, with demo JSON payloads under `config/demo/`. Tests are in `tests/` and mirror package behavior with files such as `tests/test_config_loader.py`. Utility and operational scripts live in `scripts/`. Model weights are stored in `models/` and training artifacts in `pklot_yolov84/`. Runtime outputs such as logs, frames, and SQLite data belong in `runtime/`.

## Build, Test, and Development Commands
Install dependencies with `python -m pip install -r requirements.txt`. Run the bot locally with `python -m parking_bot`, or start demo mode with `python -m parking_bot --demo`. Validate setup using `python scripts/validate_setup.py --demo` or `python scripts/validate_setup.py`. Run offline smoke checks with `python scripts/run_self_checks.py`. Execute the test suite with `pytest -q`; for focused work, use a file target such as `pytest -q tests/test_state_machine.py`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public functions, and `from __future__ import annotations` in modules that already use it. Use `snake_case` for modules, functions, variables, and test names; use `PascalCase` for classes and dataclasses. Keep imports grouped by standard library, third-party, and local package. No formatter or linter config is checked in, so match the surrounding code and keep changes small and readable.

## Testing Guidelines
Use `pytest` for all new tests. Add tests under `tests/` with filenames matching `test_*.py` and test functions named `test_*`. Prefer isolated tests built on `tmp_path` for config and repository behavior, following the existing suite. When changing runtime flows, add or update a focused `pytest` test and run `python scripts/run_self_checks.py` for the offline integration path.

## Commit & Pull Request Guidelines
The visible history is minimal and currently includes a placeholder-style subject (`Test commit`). Do better: use short imperative commit titles such as `Add demo camera config validation`. Keep pull requests narrow, describe user-visible behavior, list config or model changes, and include sample output or screenshots when Telegram flows or detection overlays change.

## Security & Configuration Tips
Create `.env` from `.env.example` and keep real tokens out of git. Treat `config/cameras.yaml`, `runtime/`, and model files as environment-specific assets; avoid committing generated logs, local databases, or temporary frames unless they are intentional fixtures.
