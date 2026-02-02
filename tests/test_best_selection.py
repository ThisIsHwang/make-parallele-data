from synth_parallel.stages import score_select_best
from synth_parallel.utils.io import write_jsonl


def test_score_select_best(tmp_path):
    run_dir = tmp_path
    selected = [
        {
            "source_id": "s1",
            "source_text": "hello",
            "length_bucket_id": 0,
            "segment_type": "sentence",
            "madlad": {},
        }
    ]
    candidates = [
        {"source_id": "s1", "translations": ["hi", "hello world"]},
    ]
    write_jsonl(str(run_dir / "selected_sources.jsonl"), selected)
    write_jsonl(str(run_dir / "candidates_128.jsonl"), candidates)

    cfg = {
        "metricx": {
            "backend": "dummy",
            "checkpoint": "dummy",
            "batch_size": 4,
            "device": "cpu",
            "device_id": None,
            "cache_db": str(run_dir / "cache.sqlite"),
            "prompt_template": "",
            "max_new_tokens": 4,
        },
        "final_generation": {"store_top_k": 1},
    }

    output = score_select_best.run(cfg, str(run_dir))
    assert output.endswith("selected_best.jsonl")
    with open(output, "r", encoding="utf-8") as f:
        line = f.readline()
    assert "metricx_qe_score_best" in line
