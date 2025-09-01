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

# 3) Create a file named key.env in the project root:
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx

# 4) Building the FAISS index:
python build_db.py

# 5) Querying the index:
python query.py "When did world war 2 begin?"

Sample Output:
─────────────────────────────────────── RAG QUERY ────────────────────────────────────────
Query Time: 2025-09-01 20:04:21 | Model: gpt-4o-mini | Embeddings: sentence-
transformers/all-MiniLM-L6-v2 | k=4

Q: When did world war 2 begin?
──────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────────────────── ANSWER ─────────────────────────────────────────
World War II began on September 1, 1939.
──────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────── SOURCES (ranked by similarity) ─────────────────────────────
1. World War II.docx
   score: 0.8441
   text:  World War II: A Global Conflict That Reshaped the Modern World World War II,
fought between 1939 and 1945, remains the most devastating conflict in human history. It
involved more than thirty countries across Europe, Asia, Africa, and the Americas,
resulting in the deaths of an estimated seventy to eighty-five million people—about three
percent of the world’s population at the time. Unlike earlie…
2. World War II.docx
   score: 0.8854
   text:  The immediate spark of the war occurred on September 1, 1939, when Germany
invaded Poland. This act of aggression prompted Britain and France to declare war on
Germany, marking the official start of World War II. The German military employed a
strategy known as Blitzkrieg, or “lightning war,” which combined rapid movements of tanks,
infantry, and aircraft to overwhelm opponents quickly. Poland fe…
3. World War II.docx
   score: 0.9460
   text:  In conclusion, World War II was not simply a war of battles and generals but a
transformative event that redefined humanity’s relationship with power, technology, and
morality. It demonstrated the extremes of human cruelty but also the capacity for courage,
sacrifice, and renewal. Its legacy continues to shape international relations, cultural
memory, and the pursuit of peace in the modern era. B…
4. World War II.docx
   score: 1.0296
   text:  The roots of World War II lay in the unresolved tensions of World War I and the
harsh terms of the Treaty of Versailles, which ended that earlier conflict in 1919.
Germany was burdened with heavy reparations, territorial losses, and severe restrictions
on its military. These conditions fostered widespread resentment, humiliation, and
economic instability. The global Great Depression of the 1930s …
