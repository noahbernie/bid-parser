# bid-parser

A PDF parsing pipeline for road construction bid documents. Extracts street work segments, cross streets, work types, and geocoded locations using Google Document AI, Gemini Flash (page screening), and Gemini Pro (street extraction).

## Local Setup

### Prerequisites

- Python 3.11+
- Node 22.12.0 (see `.nvmrc`)
- A Google Cloud project with Document AI enabled
- API keys for Gemini, Google Maps, and Anthropic

### Clone and configure

```bash
git clone https://github.com/noahbernie/bid-parser.git
cd bid-parser
cp .env.example .env
# Fill in all values in .env
```

### Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the parser locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Submit a PDF:

```bash
curl -X POST http://localhost:8000/parse \
  -H "X-Api-Key: your-parser-api-key" \
  -F "file=@/path/to/bid.pdf"
```

Poll for results using the returned `job_id`:

```bash
curl http://localhost:8000/parse/{job_id} \
  -H "X-Api-Key: your-parser-api-key"
```

### Run the eval dashboard (local only)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
open http://localhost:8000/eval
```

The eval dashboard compares parser output against ground truth CSVs in `tests/ground_truth/`.

### Run tests

```bash
# No automated test suite yet — use the eval dashboard at /eval
```

## Deployment

Deployed on Railway via Docker. Push to `main` triggers a redeploy automatically.

```bash
git push origin main
```
