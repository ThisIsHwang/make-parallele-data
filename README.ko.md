# synth-parallel (한국어 안내)

MADLAD-400에서 TranslateGemma 방식의 합성 병렬 데이터를 생성하는 파이프라인입니다.

구성:
- **Qwen teacher**: OpenAI-compatible API로 제공 (외부 서버 또는 로컬 vLLM)
- **MetricX-24 XXL QE**: 로컬 Python 코드로 GPU 추론 (서버 불필요)

---

## 1) 클론

```bash
git clone <your-repo>
cd make-data
```

---

## 2) 환경 값 준비

`.env` 및 config에 아래 값을 채워주세요.

| 항목 | 예시 | 위치 |
|---|---|---|
| Qwen API base URL | `http://qwen.myhost:8000/v1` | `configs/h100x8.yaml` -> `teacher.base_url` |
| Qwen 모델명 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | `configs/h100x8.yaml` -> `teacher.model` |
| API 키 | `your-key` | `.env` -> `VLLM_API_KEY` |
| 샤드 수 | `8` | `.env` -> `SHARDS` |

---

## 3) .env 설정

```bash
cp .env.example .env
# edit .env
```

권장 항목:
- `VLLM_API_KEY`: Qwen OpenAI-compatible API 키
- `CONFIG`: config 경로 (기본 `configs/h100x8.yaml`)
- `SHARDS`: 샤드 개수 (기본 8)
- `SKIP_VLLM=1`: Qwen이 외부 서버일 때 로컬 vLLM 실행 생략
- `TEACHER_BASE_URL`: `SKIP_VLLM=1`일 때 필수

---

## 4) uv 설치 + 의존성 설치 (단일 venv)

uv 설치 (없다면):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

uv venv 생성 및 의존성 설치:
```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
./scripts/install_metricx_official.sh
```

**주의**: MetricX 공식 requirements는 `transformers==4.30.2`, `datasets==2.13.1`로 고정되어 있어
환경이 다운그레이드될 수 있습니다.

---

## 5) MetricX (로컬 Python, GPU)

MetricX-24 공식 코드(`metricx24.models.MT5ForRegression`)를 **파이썬으로 직접 호출**합니다.
(QE 모드, 서버 불필요)

기본 설정:
- backend: `official_python`
- model: `google/metricx-24-hybrid-xxl-v2p6`
- tokenizer: `google/mt5-xl`

공식 repo 설치:
```bash
./scripts/install_metricx_official.sh
```

### 옵션: MetricX 전용 venv 분리

```bash
./scripts/install_metricx_official_venv.sh
```

그리고 config 변경:
```yaml
metricx:
  backend: official_cli
  repo_path: ./third_party/metricx
  python_bin: ./.metricx-venv/bin/python
```

---

## 6) Qwen teacher (외부 API)

외부 Qwen 서버를 사용할 경우:
```yaml
teacher:
  base_url: http://<YOUR_QWEN_HOST>:8000/v1
  api_key_env: VLLM_API_KEY
  model: Qwen/Qwen3-235B-A22B-Instruct-2507
```

키 설정:
```bash
export VLLM_API_KEY="your-key"
```

로컬 vLLM을 쓰려면:
```bash
export VLLM_API_KEY="token-abc123"
./scripts/launch_vllm_h100x8.sh
```

---

## 7) 전체 파이프라인 실행 (한 번에)

```bash
./scripts/run_sharded_pipeline.sh configs/h100x8.yaml
```

실행 순서:
1) sample_sources
2) prefilter_score (샤딩)
3) select_sources
4) generate_128 (샤딩)
5) score_select_best (샤딩)
6) format_filter (샤딩)
7) export

---

## 8) 수동 실행 (stage별)

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

## 9) 샤딩 실행 (단일 노드 8 GPU)

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i synth_parallel run \
    --config configs/h100x8.yaml \
    --stage prefilter_score \
    --shard-id $i --num-shards 8 --resume &
done
wait
```

`generate_128`, `score_select_best`, `format_filter`도 동일하게 샤딩 후,
마지막에 `export`를 1회 실행합니다.

---

## 10) 출력물

`run.out_dir` 하위에 저장됩니다:
- `sampled_sources.jsonl`
- `prefilter_candidates.jsonl` (또는 `prefilter_candidates.shard*.jsonl`)
- `selected_sources.jsonl`
- `candidates_128.jsonl` (또는 `candidates_128.shard*.jsonl`)
- `selected_best.jsonl` (또는 `selected_best.shard*.jsonl`)
- `filtered.jsonl` / `rejected.jsonl` (또는 shard 버전)
- `final.jsonl` (또는 `final.parquet`)
- `stats.json`

---

## 11) 중요 설정값

- `data.sample_pool_size`, `data.target_examples_total`
- `bucketing.boundaries`
- `teacher.base_url`, `teacher.model`, `teacher.max_concurrency`
- `teacher.validation.min_chars`, `teacher.validation.error_substrings`
- `final_generation.num_candidates`
- `metricx.checkpoint`, `metricx.backend`, `metricx.batch_size`, `metricx.device`
- `filters.llm_judge.enabled`

---

## Notes

- MetricX는 공식 `metricx24` 코드 기반이며, QE 점수는 낮을수록 좋습니다.
- vLLM이 GPU를 모두 쓰면 MetricX를 CPU로 돌리거나 MetricX 단계만 따로 실행하세요.
- 빠른 테스트는 `--dry-run` 또는 `--limit N` 옵션 사용.
