from synth_parallel.metricx.scorer import MetricXScorer


def test_dummy_metricx_scores(tmp_path):
    cfg = {
        "metricx": {
            "backend": "dummy",
            "checkpoint": "dummy",
            "batch_size": 4,
            "device": "cpu",
            "device_id": None,
            "cache_db": str(tmp_path / "cache.sqlite"),
            "prompt_template": "",
            "max_new_tokens": 4,
        }
    }
    scorer = MetricXScorer(cfg)
    scores = scorer.score_batch(["aaa", "bbbb"], ["aa", "bbbbbb"])
    assert scores[0] == 1.0
    assert scores[1] == 2.0
