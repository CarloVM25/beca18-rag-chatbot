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
    md("## Step 0 — Environment setup", "md-step0"),

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
    md("## Step 1 — Extract text from PDF", "md-step1"),

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
    md("## Step 2 — Tokenisation and chunking", "md-step2"),

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
            "### Why `chunk_size=800` with `overlap=100`?",
            "",
            "The `all-MiniLM-L6-v2` model generates 384-dimensional vectors optimised for semantic "
            "similarity at **paragraph scale** — chunks that are too long dilute the signal, while "
            "chunks that are too short lose context.",
            "",
            "* **800 characters** captures full regulatory articles (the Beca 18 document uses "
            "multi-sentence clauses that belong together semantically), keeping embedding quality "
            "high while keeping the total chunk count manageable.",
            "* **100 characters of overlap** (~12 % of chunk size) ensures sentences that straddle "
            "a boundary appear in both neighbouring chunks, preventing retrieval gaps at split points.",
            "* Because `all-MiniLM-L6-v2` runs entirely locally with no API limits, chunk count "
            "does not affect indexing cost.",
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
    md("## Step 3 — Embedding functions (local, sentence-transformers)", "md-step3"),

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
    md("## Step 4 — ChromaDB collection and indexing", "md-step4"),

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
    md("## Step 5 — Semantic search", "md-step5"),

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
    md("## Step 6 — Answer generation with Gemini 2.5 Flash", "md-step6"),

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
    md("## Step 7 — ipywidgets chat interface", "md-step7"),

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
