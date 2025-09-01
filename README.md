# Local RAG: FAISS + MiniLM + OpenAI (LangChain)

Small, practical RAG setup:
- Ingest `.docx`
- Chunk with sensible paragraph-first splitting
- Embed with `all-MiniLM-L6-v2`
- Store in FAISS
- Answer with OpenAI (`gpt-4o-mini` by default), using retrieved context only

## Setup

```bash
# 1) Create and activate a virtual env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate

# 2) Install deps
pip install -r requirements.txt
