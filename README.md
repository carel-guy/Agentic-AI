# Agentic AI

Practical examples from a first model call to an AI shopping assistant.

**Python 3.14 · LangChain · Streamlit · Chroma · LangSmith**

## Explore

|                                                       | Focus                                                    |
| ----------------------------------------------------- | -------------------------------------------------------- |
| [01 · Model calls](1_simple_llm_calling/)             | Connect to language models                               |
| [02 · Health analysis](2_health_analysis/)            | Analyze reports with a notebook and Streamlit app        |
| [03 · Vector databases](3_vector_db/)                 | Store and search embeddings                              |
| [04 · RAG](4_rag_basics/)                             | Answer questions using PDF content                       |
| [05 · Agents](5_single_agent/)                        | Query products with tools                                |
| [06 · Memory](6_memory/)                              | Keep context across conversations                        |
| [07 · Multimodal](7_multimodal/)                      | Work with images and text                                |
| [08 · Guardrails](8_guardrails/)                      | Redact emails and mask credit cards                      |
| [09 · Evaluation](9_eval/)                            | Evaluate an inventory agent with LangSmith               |
| [10 · Shopping assistant](10_project_shopping_agent/) | Search products, compare ratings, and record demo orders |

## Setup

With Python 3.14 and `uv` installed, run from the repository root:

```sh
uv sync
```

Create a `.env` file in the repository root. Fill in the keys for the examples
you want to use:

```dotenv
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
LANGSMITH_API_KEY=your_langsmith_key
```

Groq powers the agents, Google powers the health analysis examples, and
LangSmith stores evaluation results. `.env` is excluded from Git.

## Run

Run these commands individually from the repository root:

```sh
# Browse the notebooks
uv run jupyter notebook

# Try the PII guardrails demo
uv run python 8_guardrails/guardrails.py

# Query the inventory agent
uv run python 9_eval/inventory_agent.py

# Run evaluation and upload results to LangSmith
uv run python 9_eval/func_eval.py

# Launch the shopping assistant
uv run streamlit run 10_project_shopping_agent/app.py
```

The shopping app opens at **http://localhost:8501**. Its SQLite database and
sample product images are included.
