# Multi-LLM Evaluation Arena

Work in progress. The full README is written in the last phase.

## Development quick start

Requires Python 3.12+ and a running [Ollama](https://ollama.com) with at least one model
(`ollama pull llama3.2`).

## Brief demonstration
https://github.com/user-attachments/assets/a88f321e-cb05-4036-ba3c-b6ef032358a8


```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env               # Windows: copy .env.example .env

# smoke-test the Ollama connection without the web server
python -m app.cli health
python -m app.cli models
python -m app.cli ask llama3.2 "Explain TCP in two sentences"

## in my case (with 3 models llama3.2; qwen2.5:1.5b; phi3:mini)
python -m app.cli compare "Explain TCP in two sentences" llama3.2 qwen2.5:1.5b phi3:mini --judge phi3:mini

# run the API (http://localhost:8000/docs)
uvicorn app.main:app --reload

pytest
```
