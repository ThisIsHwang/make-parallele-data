from synth_parallel.data import madlad


def test_split_text_list_field():
    cfg = {
        "segmentation": {
            "mode": "auto",
            "min_chars": 2,
            "max_chars": 100,
            "merge_short": False,
            "allow_html": True,
        },
        "final_generation": {"blob": {"blob_max_tokens": 50}},
        "data": {"madlad_split": "clean", "src_lang": "kor"},
    }
    segments = madlad.segment_text_field(["hello", "world"], cfg)
    assert segments == ["hello", "world"]


def test_split_text_string_field():
    cfg = {
        "segmentation": {
            "mode": "auto",
            "min_chars": 2,
            "max_chars": 100,
            "merge_short": True,
            "allow_html": True,
        },
        "final_generation": {"blob": {"blob_max_tokens": 50}},
        "data": {"madlad_split": "clean", "src_lang": "kor"},
    }
    text = "Hello world. This is a test.\nShort."
    segments = madlad.segment_text_field(text, cfg)
    assert any("Hello world" in s for s in segments)
