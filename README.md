# GraMM-RAG — MMLongBench-Doc End-to-End Pipeline

Graph-augmented Multimodal RAG with Reward-Based Self-Healing Retrieval,
benchmarked on **MMLongBench-Doc** (1,091 questions, 135 long PDFs).

**Model:** Qwen2.5-VL-72B-Instruct (Together.ai) | **Metrics:** ANLS · F1 · Accuracy

---

## Repo structure

```
gramm-rag-mmlongbench/
├── mmlongbench_e2e.ipynb   ← main notebook (run this)
├── gen_notebook.py         ← regenerates the notebook from source
├── download_pdfs.py        ← downloads the 135 PDFs from HuggingFace
├── src/                    ← pipeline Python modules (imported by notebook)
│   ├── parsing/
│   ├── graph/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
├── data/mmlongbench/
│   └── mmlongbench_doc.json  ← 1,091 QA records (included)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

> **PDFs are not in the repo** (GitHub 100 MB file limit; total ~2 GB).  
> Run `python download_pdfs.py` after setup to fetch them from HuggingFace automatically.

---

## Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | 16 GB VRAM | A100 80 GB |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB free | 100 GB |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| CUDA | 11.8 | 12.4 |

---

## Step 1 — Clone

```bash
git clone https://github.com/wasiu-creator/gramm-rag-mmlongbench.git
cd gramm-rag-mmlongbench
```

## Step 2 — Set API keys

Create a `.env` file in the repo root (Docker Compose reads it automatically):

```bash
cp .env.example .env
nano .env          # or use any text editor
```

Fill in your keys:

```
TOGETHER_API_KEY=your-together-ai-key-here    # required — get at together.ai
OPENAI_API_KEY=your-openai-key-here           # optional — enables KG triplet extraction
```

`.env` is listed in `.gitignore` — your keys will never be committed.

> **Bare Python only:** if running without Docker, the notebook reads the same keys  
> from environment variables. Set them in your shell before launching Jupyter:
> ```bash
> export TOGETHER_API_KEY="your-together-key"
> export OPENAI_API_KEY="your-openai-key"
> ```

## Step 3 — Install dependencies

**Option A — Docker (recommended):**

```bash
docker compose build
docker compose up -d
# JupyterLab available at http://<server-ip>:8888
```

**Option B — Bare Python:**

```bash
python3.11 -m venv venv && source venv/bin/activate

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu124.html

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Step 4 — Download PDFs

```bash
python download_pdfs.py
# Downloads ~2 GB from HuggingFace into data/mmlongbench/pdfs/
# Safe to re-run — skips files already present
```

## Step 5 — Run the notebook

Open `mmlongbench_e2e.ipynb` in JupyterLab:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser   # bare Python only
```

Then: **Kernel → Restart Kernel and Run All Cells**

The notebook runs all 10 phases end-to-end (~3–6 hours on A100 80 GB).

---

## Pipeline phases

| Phase | Description | Key output |
|---|---|---|
| 0 | Setup, directories, API keys | — |
| 1–1.5 | Load 1,091 QA records + EDA charts | `results/figures/` |
| 2 | Verify PDFs in `data/mmlongbench/pdfs/` | — |
| 3 | Parse PDFs with MinerU | `parsed/mmlongbench/` |
| 4 | E5-Mistral + SigLIP embeddings → PyG graphs | `embeddings/` · `graphs/` |
| 5 | Train HGT (50 epochs, InfoNCE) | `results/models/hgt_mmlb/` |
| 6 | Fine-tune DeBERTa-v3 router (3 epochs) | `results/models/router_mmlb/` |
| 7 | Reward function grid search | `results/models/reward_mmlb.json` |
| 8 | Flat-vector RAG baseline (FAISS + cross-encoder) | `results/baseline_vector_mmlb.json` |
| 9 | GraMM-RAG (single seed: route → fuse → graph-page-guide → self-heal → gen) | `results/gramm_mmlongbench_s42.json` |
| 10 | Tables, APPA, paired bootstrap, conversion funnel, figures | `results/summary_mmlb.json` · `results/analysis_mmlb.json` · `results/analysis_extended_mmlb.json` |

**Estimated API cost:** ~$8 (Qwen2.5-VL-72B × single seed × 1,091 questions)

This notebook mirrors the **executed/proven MP-DocVQA pipeline**: cross-encoder
reranking, true-hybrid fusion, graph-guided page selection, generation-stage
self-healing, config-aware result caching (`RUN_CONFIG`), and the full
paired-bootstrap + conversion-funnel analysis suite — so MMLongBench results merge
cleanly into the joint cross-benchmark dissertation. Single seed matches the
deterministic (temperature=0, fixed-weights) config; rigour comes from the paired
bootstrap, not seed averaging.

---

## Pilot mode (quick smoke-test)

In Phase 0 of the notebook, set:

```python
N_PILOT = 100   # runs ~100 questions in ~30 min; set None for full run
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[GENERATION_ERROR]` | `TOGETHER_API_KEY` not set |
| `0 PDFs found` | Run `python download_pdfs.py` |
| `CUDA out of memory` | Reduce `text_batch_size` in Phase 4 |
| Phase 3 says `pip install magic-pdf` | Run `pip install magic-pdf[full]` |
| Port 8888 unreachable | SSH tunnel: `ssh -L 8888:localhost:8888 user@<server-ip>` |
