# Multi-LLM Evaluation Arena

Work in progress. The full README is written in the last phase.

## Development quick start

Requires Python 3.12+ and a running [Ollama](https://ollama.com) with at least one model
(`ollama pull llama3.2`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env               # Windows: copy .env.example .env

# smoke-test the Ollama connection without the web server
python -m app.cli health
python -m app.cli models
python -m app.cli ask llama3.2 "Explain TCP in two sentences"

# run the API (http://localhost:8000/docs)
uvicorn app.main:app --reload

pytest
```
