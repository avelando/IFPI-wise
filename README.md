# IFPI-wise

**Retrieval-Augmented Generation for Question Answering over Brazilian Portuguese Institutional Normative Documents**

This repository contains the reproducible experimental pipeline used in the paper *"Retrieval-Augmented Generation for Question Answering over Brazilian Portuguese Institutional Normative Documents"*, submitted to **ENIAC 2026**. The study evaluates a standard RAG pipeline over a closed collection of IFPI (Federal Institute of Piauí) normative and institutional documents, comparing lexical and dense retrieval strategies (with and without reranking) across two generator scales (`gemma3:12b` and `gemma3:27b`).

The code is **not** a proposal of a new RAG architecture. It is a diagnostic, practical study that measures the behavior, strengths, and failure modes of established retrieval and generation components in an institutional public-information scenario.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [How to Run](#how-to-run)
  - [Step 1 — Build the Index](#step-1--build-the-index)
  - [Step 2 — Run the RAG Experiments](#step-2--run-the-rag-experiments)
- [Configuration](#configuration)
- [Retrieval Methods Evaluated](#retrieval-methods-evaluated)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Notes and Known Path Conventions](#notes-and-known-path-conventions)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Overview

The system answers factual questions about IFPI normative documents (resolutions, ordinances, institutional regulations) in Brazilian Portuguese. It works in two clearly separated stages:

1. **Indexing (run once):** PDFs are cleaned, hierarchically chunked, embedded with `BAAI/bge-m3`, and stored in a FAISS vector index plus a parent document store.
2. **Experimentation (run repeatedly):** For each question, the pipeline retrieves candidate passages, optionally reranks them, builds a bounded context, generates an answer with a Gemma 3 model served by Ollama, and computes lexical, semantic, and retrieval metrics across multiple seeded runs.

All documents and questions are processed **directly in Brazilian Portuguese**, with no translation or cross-lingual adaptation.

---

## Pipeline Architecture

```
Stage 1 — Indexing (build_index)
┌─────────────────────────────────────────────────────────────┐
│  PDFs (documentos/)                                          │
│        │                                                     │
│        ▼                                                     │
│  Cleaning  → remove header/footer margins, reorder blocks    │
│        │                                                     │
│        ▼                                                     │
│  Hierarchical chunking → parent (1500/200) + child (400/50)  │
│        │                                                     │
│        ▼                                                     │
│  Embedding (BAAI/bge-m3) → FAISS index + parent docstore     │
└─────────────────────────────────────────────────────────────┘

Stage 2 — Experimentation (rag_gemma3_12b / rag_gemma3_27b)
┌─────────────────────────────────────────────────────────────┐
│  questions.json (20 QA)  ×  10 seeded rounds                 │
│        │                                                     │
│        ▼                                                     │
│  Retrieval → Dense | Dense+Rerank | BM25 | TF-IDF            │
│        │                                                     │
│        ▼                                                     │
│  Context (top-4 passages) → Gemma 3 via Ollama              │
│        │                                                     │
│        ▼                                                     │
│  Evaluation → ROUGE, BLEU, BERTScore, Token-F1, EM,          │
│               Hit@k, MRR, latency → CSV / JSON / TXT         │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

| Path | Description |
|------|-------------|
| `build_index.py` / `build_index.ipynb` | **Stage 1.** Reads and cleans the PDFs, builds hierarchical chunks, embeds child chunks, and saves the FAISS index, the parent docstore, and the index configuration. |
| `rag_gemma3_12b.py` / `rag_gemma3_12b.ipynb` | **Stage 2** using the `gemma3:12b` generator (Small Language Model setting). |
| `rag_gemma3_27b.py` / `rag_gemma3_27b.ipynb` | **Stage 2** using the `gemma3:27b` generator (Large Language Model setting). |
| `documentos/` | Input collection of institutional normative PDF documents. |
| `outputs_experiments (gemma312b)/` | Stored experiment outputs for the 12B generator. |
| `outputs_experiments (gemma327b)/` | Stored experiment outputs for the 27B generator. |
| `LICENSE` | MIT license. |
| `README.md` | This file. |

> **`.py` and `.ipynb` are equivalent.** The `.ipynb` notebooks were exported to `.py` scripts; they contain the same logic. Use whichever fits your workflow (notebooks for interactive exploration in Colab, scripts for headless execution on a server).

---

## Requirements

**Hardware (reference setup used in the paper):**
- Intel Core i5-10400F, 64 GB RAM, SSD
- NVIDIA RTX 3060 GPU with 12 GB VRAM

A CUDA-capable GPU is strongly recommended for embedding and generation. The code automatically falls back to CPU if CUDA is unavailable, but this will be considerably slower.

**Software:**
- Python 3.10+
- [Ollama](https://ollama.com/) running locally (or reachable over the network) to serve the Gemma 3 models
- The Gemma 3 models pulled into Ollama:
  ```bash
  ollama pull gemma3:12b
  ollama pull gemma3:27b
  ```

**Python packages:**

Stage 1 (indexing):
```bash
pip install langchain langchain-community sentence-transformers faiss-cpu pymupdf tqdm pandas
```

Stage 2 (experimentation):
```bash
pip install torch evaluate pandas tqdm sentence-transformers ollama \
            langchain langchain-community langchain-huggingface \
            langchain-text-splitters langchain-classic \
            rouge_score bert_score sacrebleu
```

> The evaluation stack uses Hugging Face `evaluate` with ROUGE, BLEU, and BERTScore. BERTScore uses `distilbert-base-multilingual-cased` with `lang="pt"`.

---

## Installation

```bash
git clone https://github.com/avelando/IFPI-wise.git
cd IFPI-wise
# (optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
# install the dependencies for both stages (see Requirements)
```

Start Ollama and make sure the models are available before running Stage 2:
```bash
ollama serve                    # if not already running
ollama pull gemma3:12b
ollama pull gemma3:27b
```

---

## Data Preparation

1. Place the institutional normative PDF documents inside the `documentos/` folder.
2. Provide an evaluation dataset named `questions.json` in the repository root. It must be a JSON list where **each item contains exactly these three mandatory fields**:

   ```json
   [
     {
       "documento": "resolucao_236_2025.pdf",
       "pergunta": "Qual resolução foi revogada pela Resolução Normativa nº 236 de 2025?",
       "resposta_esperada": "Foi revogada a Resolução Normativa nº 133/2022 - CONSELHO SUPERIOR, de 29 de abril de 2022."
     }
   ]
   ```

   | Field | Meaning |
   |-------|---------|
   | `documento` | File name of the reference/target document for the question. |
   | `pergunta` | The natural-language question (Brazilian Portuguese). |
   | `resposta_esperada` | The document-grounded expected answer. |

   The Stage 2 scripts validate that all three fields are present in every item before execution and will raise an error otherwise.

---

## How to Run

> **Order matters. Always run Stage 1 (build the index) first, then Stage 2 (the model scripts).** Stage 2 loads the searchable index produced by Stage 1; it will not work without it.

### Step 1 — Build the Index

Run once to preprocess the PDFs and build the searchable representations:

```bash
python build_index.py
```

This will:
- Clean the old index directories and recreate them.
- Read every PDF in `documentos/` page by page, remove upper/lower margin blocks (headers, footers, administrative marks), reorder the remaining blocks by position, and merge them into a continuous text per document.
- Preserve per-document metadata (`source`, `file_name`, `page_count`, `char_count`).
- Build a page-length distribution table (saved to `distribuicao_paginas.csv`).
- Split each document hierarchically: **parent** chunks of 1,500 characters (200 overlap) and **child** chunks of 400 characters (50 overlap).
- Embed the child chunks with `BAAI/bge-m3` and store them in a FAISS index, keeping the parent chunks for context reconstruction.
- Save the index, the docstore, and `index_config.json`.

You can also open `build_index.ipynb` and run it interactively (e.g., in Google Colab connected to a local runtime).

### Step 2 — Run the RAG Experiments

After the index exists, run either generator script:

```bash
# Small Language Model setting
python rag_gemma3_12b.py

# Large Language Model setting
python rag_gemma3_27b.py
```

Each script:
- Loads the FAISS vectorstore, the parent docstore, and the index configuration.
- Rebuilds BM25 and TF-IDF retrievers over the same parent documents, so lexical and dense retrieval share the same underlying representation.
- Loads the cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Connects to Ollama and verifies the target model is available.
- Loads and validates `questions.json`.
- Executes **10 independent rounds** (each with a different seed derived from a base seed of 42), evaluating **all four retrieval methods** on **all questions** in each round.
- With 20 questions × 4 methods × 10 rounds × 2 generators, the full protocol produces **1,600 answer generations**.
- Computes all metrics and writes results, summaries, and a final configuration snapshot to disk (partial CSVs are saved after each round for traceability).

The only difference between the two scripts is the generator: `gemma3:12b` vs `gemma3:27b`. The retrieval pipeline is identical, so retrieval-oriented metrics are the same across both.

---

## Configuration

Stage 2 behavior is controlled by the `ExperimentConfig` dataclass at the top of each RAG script. Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model_name` | `gemma3:12b` / `gemma3:27b` | Ollama generator model. |
| `ollama_host` | `http://127.0.0.1:11434` | Ollama endpoint (overridable via the `OLLAMA_HOST` environment variable). |
| `rounds` | `10` | Number of independent seeded rounds. |
| `base_seed` | `42` | Base random seed; round *n* uses `base_seed + n`. |
| `retriever_k` | `10` | Number of candidate passages retrieved. |
| `top_k_context` | `4` | Passages kept to build the final generation context. |
| `top_k_after_rerank` | `4` | Passages kept after reranking. |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `temperature` | `0.1` | Low-temperature (near-deterministic) generation. |
| `top_p` | `0.9` | Nucleus sampling parameter. |
| `num_ctx` | `8192` | Generation context window. |
| `bertscore_model` | `distilbert-base-multilingual-cased` | BERTScore backbone. |
| `bertscore_lang` | `pt` | BERTScore language. |

The generation prompt (in Brazilian Portuguese) instructs the model to answer **only** from the retrieved context, avoid inventing dates/names/numbers, produce a concise factual answer in a complete sentence, and explicitly abstain with *"Não encontrei essa informação no contexto fornecido."* when the context is insufficient.

---

## Retrieval Methods Evaluated

| Method | Description |
|--------|-------------|
| `dense_no_rerank` | Dense retrieval over the hierarchical FAISS index (BAAI/bge-m3), no reranking. |
| `dense_rerank` | Dense retrieval followed by cross-encoder reranking of the candidates. |
| `bm25` | Lexical BM25 baseline over the parent documents. |
| `tfidf` | Lexical TF-IDF baseline over the parent documents. |

In all settings the final context is built from up to **4** highest-ranked passages, concatenated with document-level source identifiers before being sent to the generator.

---

## Evaluation Metrics

**Answer quality (generation):**
- **ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum** — lexical overlap with the reference.
- **BLEU** — n-gram precision against the reference.
- **BERTScore (precision / recall / F1)** — semantic similarity.
- **Token-overlap F1** — precision/recall balance after normalization.
- **Exact Match (EM)** — strict auxiliary indicator.

**Retrieval effectiveness (generator-independent):**
- **Hit@k** for k ∈ {1, 3, 5, 10}.
- **Recall@k** — numerically equivalent to Hit@k here, since each question has a single target document.
- **MRR** (Mean Reciprocal Rank).
- **Source-in-final-context rate** — whether the target document appears in the final context passed to the generator.

**Efficiency:**
- **End-to-end latency** (seconds) per answer.

Results are aggregated at both the question level and the method level, with means and standard deviations across the 10 rounds.

---

## Output Files

Stage 2 writes to the results directory (`outputs_experiment` by default; the stored runs live under the `outputs_experiments (gemma312b)/` and `outputs_experiments (gemma327b)/` folders):

| File | Content |
|------|---------|
| `rag_resultados_parciais_ate_rodada_XX.csv` | Partial raw results saved after each round. |
| `rag_resultados_brutos.csv` | Full raw per-answer results across all rounds. |
| `rag_metricas_recuperacao_por_metodo.csv` | Retrieval metrics (Hit@k, MRR, source-in-context) per method. |
| `rag_metricas_por_questao.csv` | Per-answer generation metrics. |
| `rag_metricas_resumo_por_questao.csv` | Per-question metric summary (mean/std across rounds). |
| `rag_metricas_resumo_por_metodo.csv` | Per-method metric summary (mean/std). |
| `rag_run_config.json` | Snapshot of the exact `ExperimentConfig` used. |
| `rag_resumo_final.txt` | Human-readable textual summary of all tables. |

Stage 1 additionally writes `distribuicao_paginas.csv` (document page-length distribution) and `index_config.json` (chunking/embedding configuration).

---

## Notes and Known Path Conventions

- **Index paths differ between stages.** `build_index.py` saves the FAISS index to `faiss_final_index/` and the docstore to `final_docstore/`, while the RAG scripts expect them under `outputs_index/faiss_index/` and `outputs_index/parent_docstore/` (with `index_config.json` inside `outputs_index/`). If you run Stage 1 with its defaults, move/rename the produced artifacts into the layout the RAG scripts expect (or adjust the paths in either file) before running Stage 2.
- **`questions.json` is required in the repository root** for Stage 2 and is not produced by Stage 1 — prepare it manually as described in [Data Preparation](#data-preparation).
- **Source grounding is at the document-file level**, not the page or paragraph level, in the current implementation.
- The evaluation is based on automatic metrics and qualitative inspection by the authors; **no independent human evaluation** was conducted. The system is intended to assist in locating and summarizing normative content, **not** to replace formal legal or administrative verification (e.g., exact document identifiers, dates, revoked rules).

---

## Reproducibility

- The indexed representations are built once and reused across runs, making repeated evaluation stable and computationally practical.
- Every round uses a fixed, derived seed (`base_seed + round`), and Python, NumPy-adjacent, and Torch RNGs are seeded via `seed_everything`.
- The exact configuration used for a run is saved to `rag_run_config.json`.
- Because the dataset is small (20 questions), manually built, and restricted to a single institutional collection, results should be interpreted as an **exploratory case study**, not as broadly generalizable evidence across Brazilian public normative documents.

---

## Citation

If you use this repository, please cite the associated paper:

> Sousa, A. R. de, Sousa, J. C. R., Carvalho, K. de, Medeiros, M. M. P., and Moura, M. R. de. *Retrieval-Augmented Generation for Question Answering over Brazilian Portuguese Institutional Normative Documents.* ENIAC 2026.

```bibtex
@inproceedings{sousa2026ragifpi,
  title     = {Retrieval-Augmented Generation for Question Answering over Brazilian Portuguese Institutional Normative Documents},
  author    = {Sousa, Avelar Rodrigues de and Sousa, Jean Carlos Rodrigues and Carvalho, Karielly de and Medeiros, Manoel Messias Pereira and Moura, Marx Rodrigues de},
  booktitle = {Anais do Encontro Nacional de Intelig\^encia Artificial e Computacional (ENIAC)},
  year      = {2026}
}
```

---

## License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for details.
