# GJU Student Regulations Assistant

> Local experiment only. This copy has no GitHub remote and does not deploy to
> Streamlit. The submitted repository and public website remain unchanged.

This repository extends the page- and article-aware RAG assistant for
official GJU regulations. It runs Streamlit and PDF extraction locally while
OpenAI provides embeddings and answer generation, and Supabase stores metadata,
searchable text, and 768-dimensional vectors.

The experiment keeps the original retriever as a fallback and adds a first
GraphRAG-style path: vector/full-text/title results provide anchor articles,
explicit legal references are expanded locally, and Reciprocal Rank Fusion
combines all result lists.

## What is implemented

- Checksum-based PDF ingestion and version tracking
- Page-level PyMuPDF extraction with printed-page heuristics
- OpenAI vision OCR fallback for image-only pages
- English and Arabic article and heading detection
- Logical reading-order repair for visually stored Arabic PDF text
- Article-preserving chunks with token overlap only for long articles
- OpenAI `text-embedding-3-small` embeddings at 768 dimensions
- Supabase vector, full-text, and section-title search
- Reciprocal Rank Fusion across vector, full-text, title, and graph results
- Query planning for explicit articles, legal relations, and composite questions
- Local article-reference and adjacent-article graph expansion
- Legal ranking boosts and evidence-path explanations
- Original weighted reranker available through a feature flag
- OpenAI Responses API answers restricted to retrieved evidence
- Streamlit source cards, official links, and evidence inspection
- Row-level security that gives the public app read-only access to current data
- Unit tests for chunking, PDF/OCR extraction, retrieval, languages, and citations

## 1. Create the local environment

Copy `.env.example` to `.env` and fill in the credentials. The local `.env` is
ignored by Git and must never be committed:

```text
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
OPENAI_OCR_MODEL=gpt-5.6-luna
EXPERIMENTAL_GRAPHRAG=true
```

The public Streamlit app uses only the anonymous key. The service-role key is
loaded only by scripts under `scripts/`.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The copied `.env` already enables the experiment through the default setting.
Set `EXPERIMENTAL_GRAPHRAG=false` at any time to compare it with the original
weighted retriever.

## 2. Database safety

To try this local copy with the already indexed documents, do not run any SQL.
The experimental retriever uses the existing read-only anonymous Supabase access
and performs deterministic graph expansion in Python.

`database/graphrag_experiment.sql` is an optional design for a future, separate
Supabase project. It is intentionally not required and should not be applied to
the database used by the submitted public website.

For a completely new Supabase project, run the original base files in order:

Open the Supabase SQL editor and run these files in order:

1. `database/schema.sql`
2. `database/functions.sql`
3. `database/indexes.sql`

The base SQL enables `pgvector`, creates the four core tables, enables RLS, and adds
three read-only RPC search functions. Keep `EMBEDDING_DIMENSIONS=768`; changing
it requires changing the SQL vector type and regenerating all embeddings.

## 3. Configure official documents

`data/sources.json` is configured with all four files listed on the official
[GJU Laws & Regulations page](https://www.gju.edu.jo/content/laws-regulations-3492):
the English and Arabic 2026 books and both National Integrity Standards files.
Update this manifest when GJU publishes a new version.

```json
[
  {
    "title": "Official document title",
    "url": "https://www.gju.edu.jo/official-file.pdf",
    "department": "Responsible GJU department",
    "language": "en",
    "document_type": "regulation",
    "academic_year": "2026",
    "status": "current"
  }
]
```

Use `language: "ar"` for Arabic documents. Image-only pages are OCRed only when
ordinary PDF extraction returns no text.

## 4. Apply the first ingestion

```powershell
python scripts/ingest_documents.py
```

Unchanged checksums are skipped. Reindex one unchanged document only when
necessary:

```powershell
python scripts/reindex_document.py "Exact document title" --confirm
```

Reindexing builds a new hidden version first. Only after all pages, sections,
chunks, and vectors succeed does it mark the older version superseded. Normal
content updates use the same safe version-swap flow.

## 5. Run the application

```powershell
streamlit run app.py
```

Open `http://localhost:8501`. Until Supabase is configured, the interface opens
in setup mode and does not make API calls.

The sidebar shows `Local GraphRAG experiment: enabled`. Composite questions,
questions naming an article, and questions using terms such as “except,”
“according to,” or “final semester” trigger local graph expansion. Direct
single-rule questions still use the normal hybrid retrieval path.

To test from the terminal after ingestion:

```powershell
python scripts/test_question.py "Can I register extra credit hours in my final semester?"
```

To compare the original ranking with the local GraphRAG experiment on the same
question:

```powershell
python scripts/compare_retrieval.py "What is the course load for a student under warning in the final semester?"
```

This comparison performs two query-embedding calls but does not generate an
answer, so it is intended for occasional evaluation rather than normal use.

## 6. Verify locally

```powershell
python -m pytest -q
```

The unit tests do not call OpenAI or Supabase. Add verified real questions to
`tests/evaluation_questions.json` after the first official PDFs are indexed.

## Security boundaries

- `.env`, raw PDFs, processed snapshots, and virtual environments are ignored.
- `.dockerignore` also excludes secrets and local data from image build contexts.
- The browser never receives the OpenAI or service-role key.
- Students cannot trigger ingestion, deletion, or reindexing.
- Public database access is restricted by RLS to current document rows.
- Questions, retrieval candidates, evidence chunks, and output tokens are capped.
- `delete_document.py` requires the document UUID twice before deletion.

## How Codex and GPT-5.6 were used

Codex was used throughout the implementation, not only for the initial scaffold.
It helped turn the architecture into the ingestion, retrieval, generation, and
database layers; iterate on Arabic reading-order repair and OCR; create the
Supabase schema and row-level security policies; build and run the test suite;
diagnose PDF page-link behavior; configure GitHub Actions; and deploy the tested
application. The key design and safety decisions were reviewed against real
English and Arabic regulation documents and live retrieval results.

The running product uses GPT-5.6 through the OpenAI Responses API to generate
answers restricted to retrieved evidence. It is also the selective vision OCR
fallback for image-only PDF pages. `text-embedding-3-small` provides the
768-dimensional document and query embeddings used by the hybrid retrieval
pipeline.

## Experiment boundary

This local repository intentionally has no remote. Do not connect it to the
submitted GitHub repository or Streamlit app while comparing retrieval quality.
If the experiment proves better, its changes can be reviewed and migrated later
as a separate decision.
