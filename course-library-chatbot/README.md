# Course Library Chat

A local chatbot whose knowledge is a **fixed set of 18 documents** (cyber security, cryptography, web / mobile /
embedded security, secure coding, cloud, forensics, software testing, data science, blockchain, AI law and
usability). It answers questions only from those documents, cites the pages it used, and suggests questions to ask next.

- No upload, delete or file-selection features. The chat can't add or remove documents.
- Local model through Ollama (default `qwen2.5:1.5b-instruct`). Nothing is sent to the internet.
- Questions the documents don't cover get a "not in my documents" reply (the model isn't called).
- Suggested questions: starters on the home screen, a topic browser in the sidebar, and 3 related follow-ups after every answer.

## Run it (Windows)

1. Install Ollama (https://ollama.com/download/windows), then in PowerShell: `ollama pull qwen2.5:1.5b-instruct`
2. Double-click `run.bat` (creates a virtual environment, installs the packages, starts the app).
3. Open http://localhost:8501 if the browser doesn't open by itself.

Manual start: `pip install -r requirements.txt` then `python -m streamlit run chatbot_app.py`

## How it works

`build_corpus.py` read the PDFs once and wrote `data/corpus.json.gz` (about 14,600 text chunks with page numbers).
`kb.py` searches that file with BM25 keyword ranking, so there is no embedding model to download and the index builds
in a few seconds at start-up. `chatbot_app.py` sends the best passages to Ollama and streams the answer.

To change the library: put the PDFs in `knowledge/`, edit the list at the top of `build_corpus.py`, run
`pip install -r requirements-build.txt` and `python build_corpus.py`, and update `data/suggestions.json`.

Tests: `pip install pytest` then `python -m pytest tests`

## Known limits

- Keyword search: the question needs words that appear in the documents. Rephrase with technical terms if it says nothing was found.
- `computer_science_reasoning_about_systems.pdf` has a corrupted text layer on many pages (math-heavy), so only its readable parts are searchable.
- Image-only pages (diagrams, scanned pages) aren't read; no OCR.
- A 1.5B model summarises well but can slip on detail. Check the cited pages for anything important.

## Copyright

The PDFs are copyrighted books and `data/corpus.json.gz` contains their text. `.gitignore` keeps both out of Git. Don't
publish them in a public repository.
