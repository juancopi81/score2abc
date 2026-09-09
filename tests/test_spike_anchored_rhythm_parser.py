from PIL import Image, ImageDraw

from scripts.experiments import spike_anchored_rhythm_parser as spike

STAFF_LINES = [20, 40, 60, 80, 100]
SPACING = 20.0


def _staff_image(width: int = 180, height: int = 130) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for y in STAFF_LINES:
        draw.line((0, y, width - 1, y), fill="black", width=1)
    return image


def _anchor(order: int, x: float, y: float, pitch: str = "C5") -> dict:
    return {
        "order": order,
        "pitch": pitch,
        "center": {"x": x, "y": y},
    }


def test_extracts_directional_stem_run_and_side() -> None:
    image = _staff_image()
    draw = ImageDraw.Draw(image)
    draw.ellipse((66, 56, 78, 64), fill="black")
    draw.line((77, 18, 77, 60), fill="black", width=3)

    feature = spike.extract_anchor_features(image, [_anchor(1, 72, 60)], STAFF_LINES)[0]

    assert feature["stem"]["direction"] == "up"
    assert feature["stem"]["side"] == "right"
    assert feature["stem"]["up_run_staff_spacing"] >= 2.0
    assert feature["stem"]["up_run_staff_spacing"] > feature["stem"]["down_run_staff_spacing"]


def test_extracts_flag_connectivity_and_dot_evidence() -> None:
    flagged = _staff_image()
    flagged_draw = ImageDraw.Draw(flagged)
    flagged_draw.ellipse((46, 56, 58, 64), fill="black")
    flagged_draw.line((57, 18, 57, 60), fill="black", width=3)
    flagged_draw.line((58, 19, 82, 36), fill="black", width=5)

    dotted = _staff_image()
    dotted_draw = ImageDraw.Draw(dotted)
    dotted_draw.ellipse((106, 56, 118, 64), fill="black")
    dotted_draw.line((107, 60, 107, 108), fill="black", width=3)
    dotted_draw.ellipse((126, 58, 132, 64), fill="black")

    flag_feature = spike.extract_anchor_features(flagged, [_anchor(1, 52, 60)], STAFF_LINES)[0]
    dot_feature = spike.extract_anchor_features(dotted, [_anchor(1, 112, 60)], STAFF_LINES)[0]

    assert flag_feature["beam_flag"]["present"] is True
    assert flag_feature["beam_flag"]["density"] >= spike.FLAG_DENSITY_THRESHOLD
    assert dot_feature["beam_flag"]["present"] is False
    assert dot_feature["dot"]["present"] is True
    assert dot_feature["dot"]["area_staff_squared"] >= spike.DOT_AREA_THRESHOLD


def test_extracts_beam_connectivity_between_stems() -> None:
    image = _staff_image()
    draw = ImageDraw.Draw(image)
    draw.ellipse((46, 56, 58, 64), fill="black")
    draw.ellipse((86, 56, 98, 64), fill="black")
    draw.line((57, 10, 57, 60), fill="black", width=3)
    draw.line((97, 10, 97, 60), fill="black", width=3)
    draw.line((57, 10, 97, 10), fill="black", width=5)

    features = spike.extract_anchor_features(
        image,
        [_anchor(1, 52, 60), _anchor(2, 92, 60, "E5")],
        STAFF_LINES,
    )

    assert features[0]["beam_flag"]["present"] is True
    assert features[0]["beam_flag"]["row_peak_staff_spacing"] >= 1.0


def test_extracts_leading_residual_rest_without_counting_barline() -> None:
    image = _staff_image()
    draw = ImageDraw.Draw(image)
    # Eighth-rest-like residual in the leading gap.
    draw.line((48, 42, 59, 54, 51, 66, 62, 78), fill="black", width=5)
    draw.ellipse((56, 72, 64, 82), fill="black")
    # Full-height barline is deliberately too tall for the bounded rest shape.
    draw.line((8, 12, 8, 116), fill="black", width=3)
    groups = spike.group_simultaneous_heads([_anchor(1, 86, 60)], SPACING)

    rests = spike.extract_residual_rest_features(image, groups, STAFF_LINES)

    assert len(rests) == 1
    assert rests[0]["role"] == "leading"
    assert rests[0]["duration_beats"] == 0.5
    assert 45 <= rests[0]["center_x"] <= 65


def test_groups_vertically_aligned_heads_as_one_rhythmic_event() -> None:
    anchors = [
        _anchor(1, 40.0, 50.0, "C4"),
        _anchor(2, 44.0, 30.0, "E4"),
        _anchor(3, 82.0, 40.0, "G4"),
    ]

    groups = spike.group_simultaneous_heads(anchors, SPACING)

    assert len(groups) == 2
    assert groups[0]["pitches"] == ["C4", "E4"]
    assert groups[0]["center_x"] == 42.0
    assert groups[1]["pitches"] == ["G4"]


def test_meter_decoder_repairs_only_the_low_cost_half_beat() -> None:
    symbols = [
        {
            "kind": "note",
            "x": 10.0,
            "duration_beats": 1.0,
            "pitches": ["C5"],
            "duration_costs": {"0.5": 1.0, "1.0": 0.5, "1.5": 0.0, "2.0": 1.0},
        },
        {
            "kind": "note",
            "x": 30.0,
            "duration_beats": 0.5,
            "pitches": ["D5"],
            "duration_costs": {"0.5": 0.0, "1.0": 1.0, "1.5": 1.0, "2.0": 1.0},
        },
        {
            "kind": "note",
            "x": 50.0,
            "duration_beats": 1.0,
            "pitches": ["E5"],
            "duration_costs": {"0.5": 1.0, "1.0": 0.0, "1.5": 1.0, "2.0": 1.0},
        },
    ]

    decoded, status = spike.decode_meter(
        symbols,
        expected_beats=3.0,
        allow_pickup=False,
    )

    assert status == "meter_repaired"
    assert [symbol["duration_beats"] for symbol in decoded] == [1.5, 0.5, 1]
    assert [symbol["onset_beats"] for symbol in decoded] == [0, 1.5, 2]


def test_meter_decoder_preserves_underfull_pickup() -> None:
    symbols = [
        {"kind": "note", "x": index, "duration_beats": 0.5, "pitches": ["C5"]} for index in range(5)
    ]

    decoded, status = spike.decode_meter(
        symbols,
        expected_beats=3.0,
        allow_pickup=True,
    )

    assert status == "pickup_preserved"
    assert sum(symbol["duration_beats"] for symbol in decoded) == 2.5


def test_duration_accuracy_penalizes_extra_note_group_and_joint_gate_rejects_it() -> None:
    truth = {
        "notes": [{"onset_beats": index * 0.5, "duration_beats": 0.5} for index in range(6)],
        "rests": [],
    }
    predicted_note_groups = [
        {
            "kind": "note",
            "onset_beats": index * 0.5,
            "duration_beats": 0.5,
            "note_count": 1,
        }
        for index in range(7)
    ]
    evaluation = spike.evaluate_hypothesis(
        {"rhythm_tokens": predicted_note_groups, "rests": []}, truth
    )
    aggregate = spike.aggregate_metrics([evaluation])

    assert evaluation["duration_correct"] == 6
    assert evaluation["duration_total"] == 7
    assert evaluation["duration_accuracy"] == 6 / 7
    assert evaluation["has_note_group_overproduction"] is True
    assert aggregate["duration_accuracy"] == 0.857143
    assert spike._passes_joint_metric_gate(aggregate) is False
