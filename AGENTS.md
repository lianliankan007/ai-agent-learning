# Repository Guidelines

## Project Structure & Module Organization
This repository is organized as incremental exercises. `day1/` through `day4/` contain standalone Python examples that build from basic agents to memory-aware agents. Keep new code in the relevant day folder instead of creating cross-day dependencies unless the reuse is intentional. Root files include `requirements.txt`, `readme.md`, and `memory/` notes such as `memory/learning-progress.md`. Local secrets belong in `.local/` or environment variables and should not be committed.

## Build, Test, and Development Commands
Create an environment and install dependencies with `pip install -r requirements.txt`. Run a specific exercise directly, for example `python day4/agent_with_memory.py`. For quick syntax validation, use `python -m py_compile day4/agent_with_memory.py day4/vector_retriever.py`. If you add a new module, include the exact command needed to run it in the relevant day folder README or summary file.

## Coding Style & Naming Conventions
Use 4-space indentation and follow standard Python naming: `snake_case` for functions, variables, and modules, `PascalCase` for classes, and clear filenames such as `function_calling_demo.py`. Keep functions small and readable. This repo favors instructional code, so add concise comments where logic may confuse a beginner. Do not hardcode secrets, API keys, hosts, or tokens in source files.

## Testing Guidelines
There is no formal test suite yet. When adding non-trivial logic, add lightweight tests with names like `test_<feature>.py`, preferably in a new `tests/` directory or beside the related day module if the scope is narrow. At minimum, run the target script locally and use `python -m py_compile` before submitting changes.

## Commit & Pull Request Guidelines
Recent commits use short, focused subjects such as `rag` and `api key处理`. Prefer concise, descriptive commit messages that name the changed area, for example `day4: extract vector retriever`. Pull requests should explain the goal, list affected files, note any new environment variables, and include sample commands or output when behavior changes.

## Security & Configuration Tips
Load configuration from `.env`, `.local/openai_api_key.txt`, or environment variables. Never commit real credentials, generated caches, or IDE files. Exclude transient output such as `__pycache__/` and keep reusable defaults configurable through environment variables.

## 代码添加注释
我是一名python 新手，毕竟复杂的逻辑代码需要增加注释