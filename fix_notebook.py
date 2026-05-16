"""
Writes notebooks/beca18_rag_chatbot.ipynb to disk from scratch.
Run with:  python fix_notebook.py
"""

import json
from pathlib import Path


def cell(cell_type, source, cell_id):
    base = {"cell_type": cell_type, "id": cell_id, "metadata": {}, "source": source}
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def md(source, cell_id):
    return cell("markdown", source, cell_id)


def code(source, cell_id):
    return cell("code", source, cell_id)


CELLS = [
    # ── Title ────────────────────────────────────────────────────────────────
    md(
        "# Beca 18 RAG Chatbot\n"
        "Retrieval-Augmented Generation pipeline to answer questions about the Beca 18 regulations.",
        "md-title",
    ),

    # ── Step 0 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 0 — Environment setup",
            "",
            "Installs all required packages and initialises the Gemini client.",
            "",
            "**Packages and their roles:**",
            "",
            "| Package | Role |",
            "|---|---|",
            "| `pypdf` | Extracts raw text from each PDF page |",
            "| `tiktoken` | Counts tokens using the `cl100k_base` vocabulary — a proxy for LLM context usage |",
            "| `langchain-text-splitters` | Deterministic, overlap-aware text chunking |",
            "| `sentence-transformers` | Local embedding model — fully offline, no API cost |",
            "| `chromadb` | Embedded vector store with a persistent HNSW index |",
            "| `google-genai` | Official Gemini SDK for answer generation |",
            "| `ipywidgets` | Interactive chat UI inside the notebook |",
            "",
            "The `GEMINI_API_KEY` is read from a `.env` file rather than hardcoded to keep "
            "credentials out of source control.",
        ]),
        "md-step0",
    ),

    code(
        "%pip install -q pypdf tiktoken langchain-text-splitters "
        "google-genai chromadb ipywidgets tqdm python-dotenv sentence-transformers",
        "code-install",
    ),

    code(
        "\n".join([
            "import os",
            "import re",
            "import time",
            "import importlib.metadata",
            "",
            "import pypdf",
            "import tiktoken",
            "import chromadb",
            "import ipywidgets as widgets",
            "from tqdm.notebook import tqdm",
            "from dotenv import load_dotenv, find_dotenv",
            "from sentence_transformers import SentenceTransformer",
            "from langchain_text_splitters import RecursiveCharacterTextSplitter",
            "from google import genai",
            "from google.genai import types",
            "from IPython.display import display, Markdown",
            "",
            "load_dotenv(find_dotenv())",
            'GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")',
            'assert GEMINI_API_KEY, "GEMINI_API_KEY not found in .env"',
            "",
            "client = genai.Client(api_key=GEMINI_API_KEY)",
            "",
            "PACKAGES = [",
            '    "pypdf", "tiktoken", "langchain-text-splitters",',
            '    "google-genai", "chromadb", "ipywidgets", "tqdm",',
            '    "python-dotenv", "sentence-transformers",',
            "]",
            'print("Package versions:")',
            "for pkg in PACKAGES:",
            "    try:",
            '        print(f"  {pkg}: {importlib.metadata.version(pkg)}")',
            "    except importlib.metadata.PackageNotFoundError:",
            '        print(f"  {pkg}: not found")',
        ]),
        "code-imports",
    ),

    # ── Step 1 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 1 — Extract text from PDF",
            "",
            "Reads every page of the Beca 18 regulations PDF and produces a single cleaned string.",
            "",
            "**Design decisions:**",
            "",
            "- **`[PAGE N]` markers** are injected before each page's text so the LLM can cite "
            "exact page numbers in its answers.",
            "- **Double-space collapse** (`re.sub(r\" {2,}\", \" \", ...)`) removes the multiple "
            "spaces that PDF extractors insert around column separators and footnote markers.",
            "- **Soft-newline join** (`re.sub(r\"(?<!\\n)\\n(?!\\n)\", \" \", ...)`) merges lines "
            "that belong to the same paragraph (single `\\n`) into one line, while preserving "
            "paragraph breaks (`\\n\\n`). This keeps regulatory articles intact and improves "
            "chunking quality downstream.",
            "- Pages are joined with `\\n\\n` so the splitter in Step 2 can use paragraph "
            "boundaries as natural split points.",
        ]),
        "md-step1",
    ),

    code(
        "\n".join([
            'PDF_PATH = "../data/beca18_reglamento.pdf"',
            "",
            "def extract_text(pdf_path: str) -> str:",
            "    pages = []",
            '    with open(pdf_path, "rb") as f:',
            "        reader = pypdf.PdfReader(f)",
            "        for i, page in enumerate(reader.pages, start=1):",
            '            raw = page.extract_text() or ""',
            r'            cleaned = re.sub(r" {2,}", " ", raw)',
            r'            cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)',
            r'            pages.append(f"[PAGE {i}]\n{cleaned.strip()}")',
            r'    return "\n\n".join(pages)',
            "",
            "full_text = extract_text(PDF_PATH)",
            'print(f"Total characters : {len(full_text):,}")',
            'print(f"Total words      : {len(full_text.split()):,}")',
        ]),
        "code-extract",
    ),

    # ── Step 2 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 2 — Tokenisation and chunking",
            "",
            "Counts the total token budget of the document, then splits the text into overlapping "
            "chunks ready for embedding.",
            "",
            "**Tokenisation sub-cell:** `cl100k_base` is the same vocabulary used by GPT-4 and is "
            "a reliable estimator for any modern LLM. Knowing the total token count tells you "
            "whether the full document would fit in a single context window (it usually does not "
            "for long regulations) and helps you reason about retrieval cost vs. context stuffing.",
        ]),
        "md-step2",
    ),

    code(
        "\n".join([
            'enc = tiktoken.get_encoding("cl100k_base")',
            "tokens = enc.encode(full_text)",
            'print(f"Total tokens (cl100k_base): {len(tokens):,}")',
        ]),
        "code-tokens",
    ),

    md(
        "\n".join([
            "### Chunking parameters — `chunk_size=400`, `chunk_overlap=60`",
            "",
            "`RecursiveCharacterTextSplitter` tries to split on `\\n\\n` first (paragraph "
            "boundary), then `\\n`, then `. `, then spaces — stopping as soon as the fragment fits "
            "within `chunk_size` characters. This keeps grammatical units together.",
            "",
            "**Why 400 characters?**  ",
            "The Beca 18 regulations use short, dense clauses (one article ≈ one or two "
            "sentences). At 800 chars the embedding model receives two unrelated articles in one "
            "vector, diluting the semantic signal. At 400 chars each chunk encodes a single "
            "coherent clause, giving the retriever a tighter match target.",
            "",
            "**Why 60 characters of overlap?**  ",
            "~15 % of chunk size. A typical clause boundary falls mid-sentence; overlap ensures "
            "that sentence appears in both the preceding and following chunk, so no evidence is "
            "silently dropped at a split point.",
            "",
            "**Why character-count, not token-count?**  ",
            "`all-MiniLM-L6-v2` has a 256-token input limit. At ~4 chars/token, "
            "400 chars ≈ 100 tokens — well within that limit while leaving room for the "
            "`[PAGE N]` marker.",
        ]),
        "md-chunk-rationale",
    ),

    code(
        "\n".join([
            'METADATA = {"document": "beca18_reglamento", "topic": "beca18", "language": "es"}',
            "",
            "splitter = RecursiveCharacterTextSplitter(",
            "    chunk_size=400,",
            "    chunk_overlap=60,",
            '    separators=["\\n\\n", "\\n", ". ", " "],',
            ")",
            "",
            "docs = splitter.create_documents([full_text], metadatas=[METADATA])",
            "",
            "avg_len = sum(len(d.page_content) for d in docs) / len(docs)",
            'print(f"Total chunks     : {len(docs):,}")',
            'print(f"Avg chunk length : {avg_len:.0f} chars")',
        ]),
        "code-chunk",
    ),

    # ── Step 3 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 3 — Embedding functions (local, sentence-transformers)",
            "",
            "Loads `all-MiniLM-L6-v2` and exposes two helpers — one for batch indexing, one for "
            "single-query retrieval.",
            "",
            "**Why this model?**",
            "",
            "- **Fully local** — no network call, no API key, no per-embedding cost.",
            "- **384-dimensional output** — small enough for fast cosine search even with thousands "
            "of chunks.",
            "- **Trained for semantic similarity** — fine-tuned on over one billion sentence pairs; "
            "it maps paraphrases and synonyms to nearby vectors, which matters when user questions "
            "use different vocabulary than the regulation text "
            "(e.g. *\"requisitos\"* ↔ *\"condiciones de postulación\"*).",
            "",
            "Two separate helpers (`embed_documents` for batches, `embed_query` for single "
            "strings) keep the calling code explicit and make it straightforward to swap the model "
            "later without touching the rest of the pipeline.",
        ]),
        "md-step3",
    ),

    code(
        "\n".join([
            'st_model = SentenceTransformer("all-MiniLM-L6-v2")',
            "",
            "def embed_documents(texts):",
            "    return st_model.encode(texts, show_progress_bar=False).tolist()",
            "",
            "def embed_query(text):",
            "    return st_model.encode([text], show_progress_bar=False)[0].tolist()",
            "",
            'print("Embedding model loaded: all-MiniLM-L6-v2 (384 dims, runs locally, no API limits).")',
        ]),
        "code-embed",
    ),

    # ── Step 4 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 4 — ChromaDB collection and indexing",
            "",
            "Embeds every chunk and stores the vectors in a persistent ChromaDB collection.",
            "",
            "**Key decisions:**",
            "",
            "- **Drop-and-recreate on every run** — the `delete_collection` guard means re-running "
            "this cell after changing chunk parameters always produces a clean, consistent index. "
            "There is no risk of stale vectors from a previous parameterisation.",
            "- **Cosine distance** (`hnsw:space: cosine`) normalises vector magnitude before "
            "comparing, so a long chunk and a short chunk that express the same idea score equally. "
            "L2 distance would unfairly penalise shorter chunks.",
            "- **Batch size 100** — balances memory usage during embedding (each batch is encoded "
            "by the CPU/GPU at once) against the number of upsert round-trips to ChromaDB.",
            "- **`PersistentClient`** — the index is written to disk at `chroma_db_beca18/` and "
            "survives kernel restarts. You do not have to re-index every time you open the "
            "notebook, only when the chunks or embeddings change.",
        ]),
        "md-step4",
    ),

    code(
        "\n".join([
            'CHROMA_PATH = "../chroma_db_beca18"',
            'COLLECTION_NAME = "beca18_reglamento"',
            "BATCH_SIZE = 100",
            "",
            "chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)",
            "",
            "try:",
            "    chroma_client.delete_collection(COLLECTION_NAME)",
            "    print(f\"Dropped existing '{COLLECTION_NAME}' collection.\")",
            "except Exception:",
            "    pass",
            "",
            "collection = chroma_client.create_collection(",
            "    name=COLLECTION_NAME,",
            '    metadata={"hnsw:space": "cosine"},',
            ")",
            "",
            "texts_all = [d.page_content for d in docs]",
            "metas_all = [d.metadata for d in docs]",
            'ids_all   = [f"chunk_{i}" for i in range(len(docs))]',
            "",
            'print(f"Indexing {len(docs)} chunks...")',
            "for start in tqdm(range(0, len(docs), BATCH_SIZE)):",
            "    end = min(start + BATCH_SIZE, len(docs))",
            "    collection.upsert(",
            "        ids=ids_all[start:end],",
            "        documents=texts_all[start:end],",
            "        embeddings=embed_documents(texts_all[start:end]),",
            "        metadatas=metas_all[start:end],",
            "    )",
            "",
            'print(f"Total documents stored in ChromaDB: {collection.count():,}")',
        ]),
        "code-chroma",
    ),

    # ── Step 5 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 5 — Semantic search",
            "",
            "Implements `semantic_search` and validates it against a sample question before "
            "adding the LLM layer.",
            "",
            "**Why test retrieval in isolation?**",
            "",
            "Retrieval quality is the ceiling on answer quality — if the right chunks are not "
            "fetched, no LLM can answer correctly. Running this step independently lets you:",
            "",
            "1. Confirm the relevant regulation text appears in the top-k results.",
            "2. Inspect cosine distances: values below **0.3** indicate strong semantic overlap; "
            "values above **0.5** suggest the question vocabulary diverges from the document "
            "and re-phrasing or re-chunking may help.",
            "3. Iterate on `chunk_size`, `chunk_overlap`, or query wording without spending "
            "Gemini API quota.",
            "",
            "The function returns a plain list of dicts (`text`, `metadata`, `distance`) so "
            "results are easy to inspect and the interface is backend-agnostic.",
        ]),
        "md-step5",
    ),

    code(
        "\n".join([
            "def semantic_search(question: str, k: int = 5) -> list[dict]:",
            "    results = collection.query(",
            "        query_embeddings=[embed_query(question)],",
            "        n_results=k,",
            '        include=["documents", "metadatas", "distances"],',
            "    )",
            "    return [",
            '        {"text": text, "metadata": meta, "distance": dist}',
            "        for text, meta, dist in zip(",
            '            results["documents"][0],',
            '            results["metadatas"][0],',
            '            results["distances"][0],',
            "        )",
            "    ]",
            "",
            "",
            "# --- Test ---",
            "sample_q = \"¿Cuáles son los requisitos para postular a la Beca 18?\"",
            "results = semantic_search(sample_q, k=5)",
            "",
            "print(f\"Top-3 results for: '{sample_q}'\\n\")",
            "for i, r in enumerate(results[:3], 1):",
            r'    m = re.search(r"\[PAGE (\d+)\]", r["text"])',
            '    page = f" (page {m.group(1)})" if m else ""',
            "    print(f\"[{i}] distance={r['distance']:.4f}{page}\")",
            '    print(r["text"][:300])',
            "    print()",
        ]),
        "code-search",
    ),

    # ── Step 6 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 6 — Answer generation with Gemini 1.5 Flash",
            "",
            "Implements `answer_with_context` and runs it against five on-topic questions and one "
            "off-topic control.",
            "",
            "**Pipeline:**",
            "1. Retrieve the top-k chunks for the question (Step 5).",
            "2. Format them as a numbered context block: "
            "`[Fragmento 1]\\n…\\n\\n---\\n\\n[Fragmento 2]\\n…`",
            "3. Send context + question to Gemini with a strict system prompt.",
            "",
            "**Design decisions:**",
            "",
            "- **System prompt grounding** — explicitly instructing the model to reply "
            "*\"El documento no contiene información sobre este tema\"* when context is "
            "insufficient prevents confident hallucination on queries not covered by the document.",
            "- **`temperature=0.1`** — near-zero temperature makes responses deterministic and "
            "factual; higher values introduce paraphrasing variation that is not useful for "
            "regulatory Q&A.",
            "- **Retry with exponential backoff** — `ServerError` is transient; waiting "
            "`2^attempt` seconds (1 s, 2 s) before retrying avoids hammering the API during "
            "brief outages.",
            "- **Per-question `try/except`** — a single failing question does not abort the "
            "evaluation loop; the error is reported inline and the loop continues.",
            "- **Top-3 chunk debug print** — shows the exact cosine distance and first 200 "
            "characters of each retrieved fragment so you can see precisely what context Gemini "
            "receives, without having to re-run retrieval separately.",
        ]),
        "md-step6",
    ),

    code(
        "\n".join([
            'GEN_MODEL = "gemini-1.5-flash"',
            "",
            "from google.genai import errors as genai_errors",
            "",
            'SYSTEM_PROMPT = """Eres un asistente experto en el reglamento de la Beca 18 del gobierno peruano.',
            "Responde EXCLUSIVAMENTE con base en los fragmentos de contexto proporcionados.",
            "Cita los números de página usando el formato [PAGE N] cuando estén disponibles en el contexto.",
            "Si el contexto no contiene información suficiente para responder, di exactamente:",
            "    'El documento no contiene información sobre este tema.'",
            'No inventes información ni uses conocimiento externo."""',
            "",
            "",
            "def answer_with_context(question: str, k: int = 5, retries: int = 3) -> dict:",
            "    chunks = semantic_search(question, k=k)",
            "    distances = [round(c['distance'], 4) for c in chunks]",
            "    print(f\"  chunks={len(chunks)} | distances={distances}\")",
            '    context_block = "\\n\\n---\\n\\n".join(',
            '        f"[Fragmento {i+1}]\\n{c[\'text\']}" for i, c in enumerate(chunks)',
            "    )",
            "    for attempt in range(retries):",
            "        try:",
            "            response = client.models.generate_content(",
            "                model=GEN_MODEL,",
            "                contents=f\"Contexto recuperado del reglamento:\\n\\n{context_block}\\n\\nPregunta: {question}\",",
            "                config=types.GenerateContentConfig(",
            "                    system_instruction=SYSTEM_PROMPT,",
            "                    temperature=0.1,",
            "                ),",
            "            )",
            '            return {"answer": response.text, "sources": chunks}',
            "        except genai_errors.ServerError as e:",
            "            if attempt < retries - 1:",
            "                wait = 2 ** attempt",
            "                print(f\"  ServerError (attempt {attempt + 1}/{retries}), retrying in {wait}s: {e}\")",
            "                time.sleep(wait)",
            "            else:",
            "                raise",
            "",
            "",
            "# --- On-topic tests ---",
            "on_topic_questions = [",
            "    \"¿Cuáles son los requisitos de elegibilidad para postular a la Beca 18?\",",
            "    \"¿Cuáles son las modalidades de la Beca 18?\",",
            "    \"¿Cuál es el monto de la subvención mensual que recibe un becario de la Beca 18?\",",
            "    \"¿Cuáles son las obligaciones del becario durante sus estudios?\",",
            "    \"¿Bajo qué condiciones se puede perder la Beca 18?\",",
            "]",
            "",
            "for q in on_topic_questions:",
            "    print(f\"\\n>>> {q}\")",
            "    try:",
            "        result = answer_with_context(q)",
            "        for i, src in enumerate(result[\"sources\"][:3], 1):",
            "            print(f\"  [Chunk {i}] dist={src['distance']:.4f} | {src['text'][:200]!r}\")",
            "        display(Markdown(f\"**Q:** {q}\\n\\n**A:** {result['answer']}\\n\\n---\"))",
            "    except Exception:",
            "        display(Markdown(f\"**Q:** {q}\\n\\n**A:** Error: API unavailable\\n\\n---\"))",
            "    time.sleep(1)",
        ]),
        "code-answer",
    ),

    code(
        "\n".join([
            "# --- Off-topic test ---",
            "off_topic_q = \"¿Cuánto cuesta un pasaje Lima-Cusco en avión?\"",
            "try:",
            "    result = answer_with_context(off_topic_q)",
            "    display(Markdown(f\"**Q (off-topic):** {off_topic_q}\\n\\n**A:** {result['answer']}\"))",
            "except Exception:",
            "    display(Markdown(f\"**Q (off-topic):** {off_topic_q}\\n\\n**A:** Error: API unavailable\"))",
        ]),
        "code-offtopic",
    ),

    # ── Step 7 ───────────────────────────────────────────────────────────────
    md(
        "\n".join([
            "## Step 7 — ipywidgets chat interface",
            "",
            "Wraps the complete RAG pipeline in an interactive notebook UI — a text input, a "
            "*k* slider, and Ask / Clear buttons — with no web server required.",
            "",
            "**`on_ask` correctness guarantee:**",
            "",
            "The function follows three strict rules to prevent the answer from rendering "
            "multiple times:",
            "",
            "1. `output_area.clear_output(wait=True)` is called **at the very top** of `on_ask`, "
            "before any other side-effect. This schedules a clear that fires on the widget's next "
            "repaint, immediately replacing any stale content.",
            "2. The API call and source-widget construction happen **outside** any "
            "`with output_area:` context, so no intermediate output leaks into the widget.",
            "3. All rendering — answer Markdown and source Accordions — happens inside "
            "**exactly one** `with output_area:` block at the end of the function.",
            "",
            "**Source fragments** are displayed as collapsed `Accordion` widgets so the user can "
            "inspect the retrieved evidence without cluttering the view. Each accordion title "
            "shows the page number and cosine distance, giving a quick retrieval-quality signal.",
            "",
            "The **k slider** (1–10) lets the user adjust retrieval breadth at runtime. Higher k "
            "provides more context to Gemini but may introduce noise if semantically distant "
            "chunks are included.",
        ]),
        "md-step7",
    ),

    code(
        "\n".join([
            "def _page_from_text(text: str) -> str:",
            r'    m = re.search(r"\[PAGE (\d+)\]", text)',
            '    return f"Page {m.group(1)}" if m else "Page unknown"',
            "",
            "",
            "question_input = widgets.Text(",
            "    placeholder=\"Escribe tu pregunta sobre la Beca 18...\",",
            "    layout=widgets.Layout(width=\"70%\"),",
            "    description=\"Pregunta:\",",
            '    style={"description_width": "70px"},',
            ")",
            "k_slider = widgets.IntSlider(",
            "    value=5, min=1, max=10, step=1,",
            "    description=\"k:\",",
            '    style={"description_width": "20px"},',
            "    layout=widgets.Layout(width=\"250px\"),",
            ")",
            "ask_button   = widgets.Button(description=\"Ask\",   button_style=\"primary\", icon=\"search\")",
            "clear_button = widgets.Button(description=\"Clear\", button_style=\"warning\", icon=\"trash\")",
            "output_area  = widgets.Output()",
            "",
            "",
            "def on_ask(_):",
            "    question = question_input.value.strip()",
            "    if not question:",
            "        return",
            "    output_area.clear_output(wait=True)",
            "    error = None",
            "    result = None",
            "    try:",
            "        result = answer_with_context(question, k=k_slider.value)",
            "    except Exception as e:",
            "        error = e",
            "    source_items = []",
            "    if result is not None:",
            "        for i, src in enumerate(result[\"sources\"], 1):",
            "            title = f\"Fragment {i} — {_page_from_text(src['text'])} (distance {src['distance']:.4f})\"",
            "            item_output = widgets.Output()",
            "            with item_output:",
            "                print(src[\"text\"][:500])",
            "            acc = widgets.Accordion(children=[item_output])",
            "            acc.set_title(0, title)",
            "            acc.selected_index = None",
            "            source_items.append(acc)",
            "    with output_area:",
            "        if error:",
            "            print(f\"Error: {error}\")",
            "        else:",
            "            display(Markdown(f\"### Respuesta\\n{result['answer']}\"))",
            "            display(widgets.VBox([widgets.HTML(\"<b>Source fragments</b>\")] + source_items))",
            "",
            "",
            "def on_clear(_):",
            "    question_input.value = \"\"",
            "    output_area.clear_output()",
            "",
            "",
            "ask_button.on_click(on_ask)",
            "clear_button.on_click(on_clear)",
            "",
            "display(widgets.VBox([",
            "    widgets.HBox([question_input, k_slider, ask_button, clear_button]),",
            "    output_area,",
            "]))",
        ]),
        "code-ui",
    ),
]

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": CELLS,
}

out_path = Path(__file__).parent / "notebooks" / "beca18_rag_chatbot.ipynb"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook written to: {out_path}")
print(f"Cells: {len(CELLS)}")
