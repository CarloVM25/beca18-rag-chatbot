# Beca 18 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the official Beca 18 scholarship regulations from the Peruvian government. The source document is `data/beca18_reglamento.pdf`, the full regulatory text that governs eligibility, modalities, obligations, subsidies, and grounds for losing the scholarship.

## Pipeline

The notebook implements a six-step pipeline. First, raw text is extracted from every page of the PDF with `[PAGE N]` markers injected so the model can cite exact pages. The text is then tokenised with `tiktoken` (cl100k_base) for budget estimation, and split into 400-character overlapping chunks (60-character overlap) using `RecursiveCharacterTextSplitter`. Each chunk is embedded locally by `all-MiniLM-L6-v2` via `sentence-transformers` — no network call, no API cost — and stored in a persistent ChromaDB collection using cosine similarity. At query time, the top-k semantically closest chunks are retrieved and sent as context to Gemini, which generates a grounded answer citing page numbers and refusing to speculate when the context is insufficient.

## Installation

**Python 3.10+ is required.**

```bash
pip install -r requirements.txt
```

Copy the environment template and add your Gemini API key:

```bash
cp .env.example .env
```

Open `.env` and set:

```
GEMINI_API_KEY=your_key_here
```

You can obtain a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Running the notebook end-to-end

Open the notebook in JupyterLab or VS Code:

```bash
jupyter lab notebooks/beca18_rag_chatbot.ipynb
```

Run cells in order from top to bottom:

| Step | Cell | What it does |
|------|------|--------------|
| 0 | Environment setup | Installs packages and initialises the Gemini client |
| 1 | PDF extraction | Reads `data/beca18_reglamento.pdf` into a cleaned string |
| 2 | Tokenisation & chunking | Counts tokens, splits text into 400-char overlapping chunks |
| 3 | Embedding model | Loads `all-MiniLM-L6-v2` locally |
| 4 | ChromaDB indexing | Embeds all chunks and writes them to `chroma_db_beca18/` |
| 5 | Semantic search | Validates retrieval with a sample question |
| 6 | Answer generation | Runs five on-topic questions and one off-topic control through Gemini |
| 7 | Chat UI | Renders the interactive ipywidgets interface |

The ChromaDB index is persisted to disk after Step 4. On subsequent runs you can skip Steps 1–4 and jump straight to Step 7 as long as the chunks and embedding model have not changed.

## Using the chat interface

After running Step 7, an interactive widget appears directly in the notebook:

- **Pregunta** — type any question about the Beca 18 regulations in Spanish.
- **k slider** — controls how many chunks are retrieved (1–10). Higher k provides more context; lower k is faster and more precise for narrow questions.
- **Ask** — submits the question and displays the generated answer with page citations.
- **Clear** — resets the input and output area.

Source fragments used to generate each answer are shown as collapsed accordions beneath the response. Each accordion header includes the page number and cosine distance, so you can inspect retrieval quality at a glance. A distance below 0.3 indicates strong semantic overlap; above 0.5 suggests the question vocabulary diverges from the document.

## Model notes

- **Embeddings** — generated locally by `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions). No internet connection or API key is required for the embedding step.
- **Generation** — handled by `gemini-2.5-flash` via the `google-genai` SDK. Requires a valid `GEMINI_API_KEY`. The model is instructed to answer strictly from retrieved context and reply *"El documento no contiene información sobre este tema."* for out-of-scope questions.
