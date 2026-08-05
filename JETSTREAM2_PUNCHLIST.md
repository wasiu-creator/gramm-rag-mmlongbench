# MMLongBench-Doc on JetStream2 — Execution Punch List

Step-by-step runbook to execute the GraMM-RAG MMLongBench-Doc pilot (and full
run) on a JetStream2 GPU instance. Mirrors the path that executed MP-DocVQA.

Repo: `wasiu-creator/gramm-rag-mmlongbench` (flat layout — `src/` sits beside the
notebook at repo root). Notebook is **generated**: edit `gen_notebook.py`, then
`python gen_notebook.py` — never hand-edit `mmlongbench_e2e.ipynb`.

Dataset: `yubo2333/MMLongBench-Doc` — 1,091 QA records, 135 PDFs (avg ~47.5
pages). QA JSON is committed (`data/mmlongbench/mmlongbench_doc.json`); the PDFs
are pulled from HuggingFace by `download_pdfs.py`.

---

## 0. Provision the instance

- [ ] Launch a JetStream2 **GPU** flavor (`g3.large` / A100 40 GB is ample;
      E5-Mistral-7B 4-bit ~4 GB, SigLIP ~1.5 GB, HGT/router tiny — 16 GB VRAM
      is the floor).
- [ ] Image: Ubuntu 22.04 + NVIDIA driver (or the JetStream2 featured
      "Ubuntu 22 + CUDA/Docker" image).
- [ ] `nvidia-smi` shows the GPU.
- [ ] NVIDIA container runtime works:
      `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
      (if it fails: install `nvidia-container-toolkit`, `sudo systemctl restart docker`).

## 1. Get the code

```bash
git clone https://github.com/wasiu-creator/gramm-rag-mmlongbench.git
cd gramm-rag-mmlongbench
```

## 2. Get the PDFs (135 files, ~2 GB — NOT in git)

The QA JSON is already in the repo. Fetch the PDFs from HuggingFace:

```bash
# Option A — download on the instance (needs network egress + huggingface_hub):
pip install huggingface_hub
python download_pdfs.py            # -> data/mmlongbench/pdfs/  (skips existing)

# Option B — copy a local set you already have, into the Docker mount target:
mkdir -p pdfs_volume
#   scp -i grammrag.pem -r /path/to/mmlongbench/pdfs/* ubuntu@<INSTANCE_IP>:~/gramm-rag-mmlongbench/pdfs_volume/
```
- [ ] PDFs present. Docker mounts `./pdfs_volume` → `/app/data/mmlongbench/pdfs`,
      so put them under `pdfs_volume/` (Option B), OR run `download_pdfs.py` once
      inside the container in Step 6 (it writes to the same mounted path).
- [ ] Sanity: `ls pdfs_volume | head` (or `ls data/mmlongbench/pdfs | head`)
      shows `*.pdf`. Expected 134–135 (one filename is very long and may be
      absent — safe to continue).

## 3. API keys

```bash
cp .env.example .env
# edit .env — set BOTH:
#   TOGETHER_API_KEY=...   (Qwen2.5-VL-72B generation)
#   OPENAI_API_KEY=...     (KG triplet extraction; optional but recommended)
```
- [ ] Keys set (no placeholders). `.env` is gitignored — never commit it.

## 4. Build + launch the container

```bash
docker compose --env-file .env up -d --build
```
- [ ] Build succeeds (first build ~10–20 min: CUDA torch + PyG + magic-pdf).
- [ ] `docker compose ps` healthy.
- [ ] Jupyter Lab reachable via SSH tunnel from your laptop:
      `ssh -i grammrag.pem -L 8888:localhost:8888 ubuntu@<INSTANCE_IP>`
      then browse `http://localhost:8888` (no token). Open `mmlongbench_e2e.ipynb`.

## 5. Configure the run (Phase 0, cell 1)

- [ ] **Pilot first:** leave `N_PILOT = 100` (fast smoke of the full pipeline).
- [ ] Proven levers are ON by default and should stay ON for the headline run:
      `USE_CROSS_ENCODER, HYBRID_FUSION, GRAPH_PAGE_GUIDE, ROUTE_ALL_HYBRID,
      SELF_HEAL_GENERATION`.
- [ ] Generator is `GEN_MODEL = 'Qwen/Qwen2.5-VL-72B-Instruct'`.

## 6. Execute

**Interactive (recommended for the pilot):** Kernel → Restart & Run All.

**Headless (recommended for the full run — survives disconnects):**
```bash
docker compose exec gramm-rag-mmlb \
  python -m nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=-1 \
    --output mmlongbench_e2e_executed.ipynb mmlongbench_e2e.ipynb
```
(If you skipped Step 2, run `python download_pdfs.py` inside the container first.)

### Expected phase behaviour / gotchas
- [ ] **Phase 2** reports ~134–135 PDFs found.
- [ ] **Phase 3 (MinerU)** is the slow part — MMLongBench PDFs average ~47.5
      pages. Budget ~30 s–2 min/doc; the ~90–100 pilot docs may take **1–2 h**
      first run. It is **cached** (`parsed/*.json`) — re-runs skip it. First doc
      also downloads MinerU model weights.
- [ ] **Phase 4** downloads E5-Mistral-7B + SigLIP on first run (a few GB), then
      caches `*_text.pt / *_text_raw.pt / *_img.pt`. Confirm `_text_raw.pt` is
      written (semantic vector retrieval depends on it).
- [ ] **Phase 8** prints `FAISS indices built (in-memory, from _text_raw.pt)`.
- [ ] **Phase 9** logs per-question `route=... R=... refused=...`.
- [ ] **Phase 10** prints the headline table (ANLS/F1/Acc/APPA/Abst-F1), the
      per-route Table A, McNemar Table B, the paired-bootstrap CI, and the
      conversion funnel.

## 7. Collect results

Everything lands in `results/` (mounted at `./cache/results` on the host):
- [ ] `summary_mmlb.json` — headline metrics + config
- [ ] `analysis_mmlb.json` — paired bootstrap + win/loss + funnel
- [ ] `analysis_extended_mmlb.json` — forest CIs / correlation / difficulty
- [ ] `gramm_mmlongbench_s42.json`, `baseline_vector_mmlb.json` — raw per-Q data
- [ ] `results/figures/*.png` — APPA, quote/route tables, bootstrap, funnel,
      forest, correlation, scatter, reward-vs-ANLS
- [ ] Copy back:
      `scp -i grammrag.pem -r ubuntu@<INSTANCE_IP>:~/gramm-rag-mmlongbench/cache/results ./mmlb_results`

## 8. Full run

Once the pilot looks right:
- [ ] Set `N_PILOT = None` in Phase 0 (full 1,091-question run; trains the
      DeBERTa router properly instead of the pilot's 1-epoch pass, and runs HGT
      for 50 epochs).
- [ ] **Clear stale results** so the config-aware cache recomputes cleanly:
```bash
docker compose exec gramm-rag-mmlb bash -lc 'rm -f results/*.json && \
  rm -rf results/models/hgt_mmlb results/models/router_mmlb \
         results/models/reward_mmlb.json'
```
  (Result JSONs also auto-invalidate when `RUN_CONFIG` changes — a full-vs-pilot
  change flips `N_PILOT` inside `RUN_CONFIG` — but clearing is safest.)
- [ ] Re-run headless (Step 6). Budget: MinerU parse of all 135 docs is the long
      pole (a few hours first time); generation ~$8 API (single seed).

## 9. Shut down

- [ ] `docker compose down`
- [ ] **Shelve or delete** the JetStream2 instance to stop burning allocation
      (`openstack server shelve <id>` or via the Exosphere/Horizon UI).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[GENERATION_ERROR]` on every question | wrong/absent `TOGETHER_API_KEY` or model | Confirm `Qwen/Qwen2.5-VL-72B-Instruct` is serverless-available on your Together account; check Phase 0 prints "API keys loaded from .env" |
| Phase 2 finds 0 PDFs | PDFs not in the mounted path | Put them under `./pdfs_volume/` (host) → `/app/data/mmlongbench/pdfs`, or run `download_pdfs.py` inside the container |
| Phase 3 slow / OOM on a long PDF | ~47-page average | it's cached per-doc; let it finish, or temporarily lower the pilot size |
| Results identical after a config change | stale cache | `RUN_CONFIG` should auto-recompute; if unsure, delete `results/*.json` |
| `magic-pdf` import error | MinerU weights not fetched | first parse downloads them; ensure network egress is open |
| KG triplets empty warning | no `OPENAI_API_KEY` | safe to proceed — the graph is still built; set the key for KG edges |
| Notebook shows old code after `gen_notebook.py` | Jupyter cached the old tab | close & reopen `mmlongbench_e2e.ipynb` AND restart the kernel |
