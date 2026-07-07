from gui_g2_verl.reward import compute_score


def test_exact_bbox_gets_high_reward():
    score = compute_score(
        data_source="unit",
        solution_str="[10,20,30,40]",
        ground_truth={"bbox": [10, 20, 30, 40], "width": 100, "height": 100},
        extra_info={},
    )
    assert score["score"] == 1.0
    assert score["point"] == 1.0
    assert score["coverage"] == 1.0
    assert score["format"] == 1.0


def test_far_bbox_gets_lower_reward():
    near = compute_score("unit", "[10,20,30,40]", {"bbox": [10, 20, 30, 40]}, {})
    far = compute_score("unit", "[70,70,90,90]", {"bbox": [10, 20, 30, 40]}, {})
    assert far["score"] < near["score"]


def test_bad_format_has_zero_score():
    score = compute_score("unit", "x=10 y=20", {"bbox": [10, 20, 30, 40]}, {})
    assert score["score"] == 0.0
