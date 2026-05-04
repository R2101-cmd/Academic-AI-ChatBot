# Academic AI Companion

Full-stack final-year engineering project: a professional Academic AI Companion with a ChatGPT-style React UI and a Python RAG backend.

## System Flow

User Query -> NLP Processing -> SentenceTransformer Embeddings -> FAISS Retrieval -> Graph-CoT Learning Path -> TinyLlama/Ollama Generation -> Verification -> Quiz/Flashcard Generation

## Main Modules

- `backend/modules/nlp.py`: query cleaning, mode detection, academic-only restriction
- `backend/modules/retrieval.py`: SentenceTransformers `all-MiniLM-L6-v2` embeddings and FAISS search
- `backend/modules/graph_cot.py`: prerequisite learning path and graph payload
- `backend/modules/generation.py`: TinyLlama through Ollama with grounded fallback
- `backend/modules/verification.py`: context and learning-path verification
- `backend/modules/quiz.py`: dynamic quiz generation
- `backend/modules/flashcard.py`: dynamic flashcard generation
- `backend/modules/personalization.py`: SQLite learner progress and difficulty

## Run

Backend:

```bash
uvicorn backend.api:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

If PowerShell says `node` is not recognized, run the frontend with Node added to the current terminal PATH:

```powershell
cd frontend
$env:Path = "C:\Program Files\nodejs;" + $env:Path
npm.cmd run dev
```

To fix it permanently on Windows, add `C:\Program Files\nodejs` to your user `Path` environment variable, then close and reopen PowerShell.

For local LLM responses, install Ollama and run:

```bash
ollama run tinyllama
```
