# synth-parallel

End-to-end pipeline to reproduce TranslateGemma-style synthetic parallel data generation from MADLAD-400, using:
- Qwen teacher served via OpenAI-compatible API (external or local vLLM)
- MetricX-24 XXL QE scoring **locally in Python** (no server required)

This README is written so you can clone the repo and run data generation in one go.
한국어 안내는 `README.ko.md`를 참고하세요.

---

## 1) Clone

```bash
git clone <your-repo>
cd make-data
```

---

## 2) Your environment (fill this in)

Fill in these values in `.env` and/or your config:

| Item | Example | Where |
|---|---|---|
| Qwen API base URL | `http://qwen.myhost:8000/v1` | `configs/h100x8.yaml` -> `teacher.base_url` |
| Qwen model name | `Qwen/Qwen3-235B-A22B-Instruct-2507` | `configs/h100x8.yaml` -> `teacher.model` |
| API key | `your-key` | `.env` -> `VLLM_API_KEY` |
| Shards | `8` | `.env` -> `SHARDS` |
| HF endpoint (optional) | `https://hf-mirror.com` | `configs/h100x8.yaml` -> `data.hf_endpoint` |
| MADLAD dataset id | `allenai/MADLAD-400` | `configs/h100x8.yaml` -> `data.madlad_dataset` |

---

## 3) Configure `.env`

```bash
cp .env.example .env
# edit .env
```

Recommended fields in `.env`:
- `VLLM_API_KEY`: API key for your Qwen OpenAI-compatible server
- `CONFIG`: config path (default: `configs/h100x8.yaml`)
- `SHARDS`: number of shards (default: 8)
- `SKIP_VLLM=1`: set if Qwen is external (don’t launch local vLLM)
- `TEACHER_BASE_URL`: required when `SKIP_VLLM=1`

---

## 4) Install uv + dependencies (single venv)

Install uv (if not installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

Create venv and install deps:
```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
./scripts/install_metricx_official.sh
```

**Important**: MetricX official requirements pin `transformers==4.30.2`, `datasets==2.13.1`. This may downgrade your environment.

---

## 5) MetricX: run locally in Python (GPU)

We use the **official MetricX-24 code** directly in Python (`metricx24.models.MT5ForRegression`), QE mode, **no server**.
Config defaults:
- backend: `official_python`
- model: `google/metricx-24-hybrid-xxl-v2p6`
- tokenizer: `google/mt5-xl`

Make sure official repo is installed:
```bash
./scripts/install_metricx_official.sh
```

### Option: separate venv for MetricX (isolation)
If you want MetricX deps isolated, install into a separate venv and switch backend:

```bash
./scripts/install_metricx_official_venv.sh
```

Then in your config:
```yaml
metricx:
  backend: official_cli
  repo_path: ./third_party/metricx
  python_bin: ./.metricx-venv/bin/python
```

This uses official `metricx24.predict` CLI in the separate venv.

---

## 6) Qwen teacher (external API)

You said you will serve Qwen externally. Set this in your config:
```yaml
teacher:
  base_url: http://<YOUR_QWEN_HOST>:8000/v1
  api_key_env: VLLM_API_KEY
  model: Qwen/Qwen3-235B-A22B-Instruct-2507
```

Then export the key:
```bash
export VLLM_API_KEY="your-key"
```

If you want local vLLM instead (optional):
```bash
export VLLM_API_KEY="token-abc123"
./scripts/launch_vllm_h100x8.sh
```

---

## 7) Run the full pipeline (one command)

```bash
./scripts/run_sharded_pipeline.sh configs/h100x8.yaml
```

This will execute all stages in order:
1) sample_sources
2) prefilter_score (sharded)
3) select_sources
4) generate_128 (sharded)
5) score_select_best (sharded)
6) format_filter (sharded)
7) export

All stages support `--resume` to continue from existing outputs.

---

## 8) Manual run (stage by stage)

```bash
synth_parallel run --config configs/h100x8.yaml --stage sample_sources
synth_parallel run --config configs/h100x8.yaml --stage prefilter_score
synth_parallel run --config configs/h100x8.yaml --stage select_sources
synth_parallel run --config configs/h100x8.yaml --stage generate_128
synth_parallel run --config configs/h100x8.yaml --stage score_select_best
synth_parallel run --config configs/h100x8.yaml --stage format_filter
synth_parallel run --config configs/h100x8.yaml --stage export
```

---

## 9) Sharding (single node, 8 GPUs)

Large stages support sharding. Example for 8 shards:

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i synth_parallel run \
    --config configs/h100x8.yaml \
    --stage prefilter_score \
    --shard-id $i --num-shards 8 --resume &
done
wait
```

Repeat for `generate_128`, `score_select_best`, `format_filter`. Then run `export` once.

---

## 10) Outputs

All outputs go under `run.out_dir`:
- `sampled_sources.jsonl`
- `prefilter_candidates.jsonl` (or `prefilter_candidates.shard*.jsonl`)
- `selected_sources.jsonl`
- `candidates_128.jsonl` (or `candidates_128.shard*.jsonl`) — one record per source with `translations: [...]`
- `selected_best.jsonl` (or `selected_best.shard*.jsonl`)
- `filtered.jsonl` / `rejected.jsonl` (or `filtered.shard*.jsonl` / `rejected.shard*.jsonl`)
- `final.jsonl` (or `final.parquet`)
- `stats.json`

---

## 11) Important config knobs

See `configs/example.yaml` / `configs/h100x8.yaml`:
- `data.sample_pool_size`, `data.target_examples_total`
- `bucketing.boundaries`
- `data.lang_map` (ex: `kor -> ko`, `eng -> en`)
- `teacher.base_url`, `teacher.model`, `teacher.max_concurrency`
- `teacher.validation.min_chars`, `teacher.validation.error_substrings` (invalid 응답 감지/재시도)
- `teacher.unset_proxies_before_request` (Qwen 호출 직전에만 proxy 해제)
- `final_generation.num_candidates`
- `metricx.checkpoint`, `metricx.backend`, `metricx.batch_size`, `metricx.device`
- `filters.llm_judge.enabled`
- `data.hf_timeout_s`, `data.hf_endpoint`, `data.hf_enable_hf_transfer` (HF timeout/미러)

---

## Server deploy helpers

- `./scripts/deploy_server.sh` uses `rsync` to push the repo to a target host.
- `./scripts/bootstrap_ubuntu.sh` installs uv + venv + base dependencies.
- `deploy/systemd/vllm.service` is a sample systemd unit for vLLM.

---

## Notes

- MetricX uses official `metricx24` code in Python: `metricx.backend: official_python`.
- Official repo is installed at `third_party/metricx` by `./scripts/install_metricx_official.sh`.
- Official requirements pin `transformers==4.30.2`, `datasets==2.13.1`.
- If you see `Dataset scripts are no longer supported` errors, ensure `datasets<3.0`.
- If vLLM uses all GPUs, either run MetricX on CPU or run MetricX stages after stopping vLLM.
- For fast iteration use `--dry-run` or `--limit N`.
