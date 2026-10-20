# 🧾 Invoice to Spreadsheet

A Streamlit app that reads invoices and receipts with Google Gemini and turns them into spreadsheets you can check, edit and download.

## Features
- Upload up to 10 invoices or receipts at once (PDF, PNG, JPG, WEBP)
- Extracts vendor, invoice number, dates, currency, line items, subtotal, tax and total
- Checks that line items, subtotal, tax and total add up, and flags anything that doesn't
- Edit any cell in the app before exporting
- Download as Excel (two sheets) or CSV
- Non-invoice files are skipped, and one bad file never stops the rest

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

- create `.streamlit/secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```
- or set an environment variable (`GEMINI_API_KEY`), or paste the key into the app sidebar.

Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | required |
| `GEMINI_MODEL` | secrets / env var | `gemini-3.6-flash` |

If the main model is busy, the app retries and then tries the backup models listed in `FALLBACK_MODELS` at the top of `app.py`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit `.streamlit/secrets.toml`). `app.py`, `requirements.txt` and `README.md` must be in the top-level folder.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo and branch, set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- AI reading can make mistakes, especially with blurry photos or handwriting. Always check the **Needs review** column and the totals before using the data for accounting.
- Files are sent to the Gemini API. Don't upload documents containing data you aren't allowed to share.
- Text in exported files that starts with `=`, `+`, `-` or `@` gets a leading `'` so a malicious invoice can't run formulas when opened in Excel.
