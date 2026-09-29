# Agentic AI

A collection of practical projects exploring language models, retrieval, tools,
memory, and evaluation — from a first API call to complete Streamlit apps.

**Python 3.14 · LangChain · Streamlit · Chroma · LangSmith**

[Explore the projects](#projects) · [Get started](#quickstart) · [Telecom chatbot](11_project_telecom_chatbot/README.md)

## Projects

Follow the numbered folders in order, or jump into a project.

| Project | What it covers | Format |
| --- | --- | --- |
| [01 · Model calls](1_simple_llm_calling/) | Connect to language model providers | Notebook |
| [02 · Health analysis](2_health_analysis/) | Extract and summarize blood work reports | Notebook + app |
| [03 · Vector databases](3_vector_db/) | Store embeddings and search similar content | Notebook |
| [04 · RAG basics](4_rag_basics/) | Answer questions using a PDF as context | Notebook |
| [05 · Tools and agents](5_single_agent/) | Query products through an agent tool | Notebook |
| [06 · Memory](6_memory/) | Maintain context across conversations | Notebook |
| [07 · Multimodal](7_multimodal/) | Combine image understanding and tools | Notebook |
| [08 · Guardrails](8_guardrails/) | Redact emails and mask credit card data | Python |
| [09 · Evaluation](9_eval/) | Compare inventory answers with reference examples | Python + LangSmith |
| [10 · Shopping assistant](10_project_shopping_agent/README.md) | Search by text or image, compare ratings, and record demo orders | Streamlit |
| [11 · Telecom chatbot](11_project_telecom_chatbot/README.md) | Retrieve answers from FAQs, resolved tickets, and a PDF guide | Streamlit + CLI |

## Quickstart

Use **Python 3.14** and **uv**. Run all commands below from the repository root,
the folder containing this README and the main `pyproject.toml`.

### 1. Install dependencies

```sh
uv sync
```

This creates the shared `.venv` used by the examples. Keep using the repository
root when running the telecom project; it also contains a separate project manifest.

### 2. Configure the providers you need

Create a `.env` file at the repository root, or update the existing file.
Add only the credentials needed for the examples you plan to run.

```dotenv
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
LANGSMITH_API_KEY=your_langsmith_key
```

| Setting | Used for |
| --- | --- |
| `GROQ_API_KEY` | Groq model calls and the agent projects |
| `GOOGLE_API_KEY` | Google model calls and health analysis |
| `LANGSMITH_API_KEY` | Uploading evaluation runs from project 09 |

The `.env` file is excluded from Git. Local embeddings do not require a Groq
key; downloading an embedding model for the first time requires internet access.

### 3. Choose an entry point

```sh
# Open the learning notebooks
uv run jupyter notebook

# Launch the shopping assistant
uv run streamlit run 10_project_shopping_agent/app.py
```

Streamlit opens at **http://localhost:8501**. Use `Ctrl+C` to stop the server.

## Run the examples

Run each command separately.

```sh
# Health analysis
uv run streamlit run 2_health_analysis/streamlit_app/app.py

# PII guardrails
uv run python -X utf8 8_guardrails/guardrails.py

# Inventory agent
uv run python -X utf8 9_eval/inventory_agent.py

# Evaluation — uploads results to LangSmith
uv run python -X utf8 9_eval/func_eval.py
```

**Shopping assistant.** Its SQLite database and sample images are included.
Start with a request such as `Show organic honey under $15`.
See the [project guide](10_project_shopping_agent/README.md) for configuration.

**Telecom chatbot.** Build the three Chroma collections before starting the app.
The [telecom guide](11_project_telecom_chatbot/README.md) covers source checks,
ingestion, retrieval tests, and both interfaces. To inspect its prerequisites:

```sh
uv run python -X utf8 11_project_telecom_chatbot/check_setup.py
```

<details>
<summary>PowerShell: use the existing virtual environment directly</summary>

From the repository root, you can run the same commands with its Python
interpreter. Environment activation is optional.

```powershell
.\.venv\Scripts\python.exe -X utf8 .\9_eval\inventory_agent.py
.\.venv\Scripts\python.exe -m streamlit run .\10_project_shopping_agent\app.py
```

</details>

## Models and local data

Projects **08–11** read `GROQ_CHAT_MODEL` from the environment. The shopping app
also reads `GROQ_VISION_MODEL`. Their configured defaults are:

```dotenv
GROQ_CHAT_MODEL=qwen/qwen3.8-27b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

These settings are optional. Other lessons keep their model configuration in
their own notebooks or scripts. Model availability can change; consult the
[Groq model list](https://console.groq.com/docs/models) when selecting a replacement.

The telecom project uses `sentence-transformers/all-MiniLM-L6-v2`, matching
project 04, and stores its index in `11_project_telecom_chatbot/chroma_store/`.
Its setup checks do not ingest data. The shopping project records demo orders
in its local `store.db`.

These are learning examples. Project 09 measures sentence similarity rather
than exact stock correctness, and project 11 treats each question independently.
