# AI Shopping Assistant

Run these commands in PowerShell from the repository root:

```powershell
cd C:\Users\Sogo\Documents\agentic-ai\Agentic-AI
.\.venv\Scripts\python.exe -m streamlit run .\10_project_shopping_agent\app.py
```

Open http://localhost:8501. Stop the server with Ctrl+C in that terminal.
Use the repository's `.venv` interpreter so that Streamlit and the agent use
the installed project dependencies. This project uses Python 3.14.

## Configuration

The repository root `.env` must contain `GROQ_API_KEY`. The current local setup
already has it configured. Do not commit this file or share the key.

Both chat and image analysis default to `qwen/qwen3.8-27b`. You can optionally
add these settings to the existing `.env` to select other Groq models:

```dotenv
GROQ_CHAT_MODEL=qwen/qwen3.8-27b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

The chat model must support tool calling; the vision model must support image
input. Check [Groq's model list](https://console.groq.com/docs/models) and
[vision documentation](https://console.groq.com/docs/vision) when changing them.
The original Qwen 3 32B and Llama 4 Scout models were retired for free/developer
accounts on July 17, 2026, per
[Groq's deprecation notice](https://console.groq.com/docs/deprecations).

## Data and services

`store.db` already contains the `products`, `reviews`, and `orders` tables.
The reviews module reads this SQLite database directly; no separate API server
or database service is needed. Orders are stored locally in this demo database.

There is no need to run `setup_db.py` for the existing database. That script
replaces the seeded products and reviews, so use it only when intentionally
initializing or reseeding the demo data.

Try `I want organic honey under $15 with 4+ rating`, or upload
`resources/honey.png` or `resources/oats.png` in the sidebar.
Chat and image search require an internet connection and available Groq quota.

## Dependency check

All direct app dependencies are already declared in the root `pyproject.toml`.
To check the existing environment:

```powershell
uv pip check --python .venv\Scripts\python.exe
```

To install the locked project dependencies on a fresh checkout, run `uv sync`
from the repository root, then configure `GROQ_API_KEY` in `.env`.
