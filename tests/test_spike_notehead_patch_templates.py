from __future__ import annotations

from PIL import Image, ImageDraw

from scripts.experiments import spike_notehead_patch_templates as spike


def test_normalized_patch_is_stable_across_staff_spacing() -> None:
    small = _scaled_notehead_image(staff_spacing=20)
    large = _scaled_notehead_image(staff_spacing=40)

    small_patch = spike._extract_normalized_patch(
        small,
        center=(40.0, 40.0),
        staff_spacing=20.0,
        kind="grayscale",
        threshold=127,
    )
    large_patch = spike._extract_normalized_patch(
        large,
        center=(80.0, 80.0),
        staff_spacing=40.0,
        kind="grayscale",
        threshold=127,
    )

    distance = spike._shift_tolerant_distance(
        small_patch,
        large_patch,
        spike.PATCH_WIDTH,
        spike.PATCH_HEIGHT,
    )
    assert distance < 0.015


def test_staff_line_suppression_removes_horizontal_ink() -> None:
    image = Image.new("L", (80, 80), 255)
    draw = ImageDraw.Draw(image)
    draw.line((0, 40, 79, 40), fill=0, width=2)

    raw = spike._extract_normalized_patch(
        image,
        center=(40.0, 40.0),
        staff_spacing=20.0,
        kind="binary",
        threshold=127,
    )
    suppressed_image = spike._suppress_staff_lines(image, staff_lines=(40,), staff_spacing=20.0)
    suppressed = spike._extract_normalized_patch(
        suppressed_image,
        center=(40.0, 40.0),
        staff_spacing=20.0,
        kind="binary",
        threshold=127,
    )

    assert sum(raw) > 0
    assert sum(suppressed) == 0


def test_shift_tolerant_distance_aligns_one_pixel_translation() -> None:
    width = 7
    height = 7
    centered = _binary_vector(width, height, {(3, 3), (4, 3), (3, 4)})
    shifted = _binary_vector(width, height, {(4, 3), (5, 3), (4, 4)})

    exact = spike._shift_tolerant_distance(centered, shifted, width, height, max_shift=0)
    tolerant = spike._shift_tolerant_distance(centered, shifted, width, height, max_shift=1)

    assert exact > 0
    assert tolerant == 0


def test_template_and_knn_scorers_rank_shifted_notehead_patch_first() -> None:
    positive_vectors = [
        _canonical_shape({(7, 6), (8, 6), (9, 6), (8, 5), (8, 7)}),
        _canonical_shape({(7, 6), (8, 6), (9, 6), (7, 7), (8, 7)}),
        _canonical_shape({(7, 5), (8, 5), (9, 5), (8, 6), (9, 6)}),
    ]
    negative_vectors = [
        _canonical_shape({(8, y) for y in range(2, 11)}),
        _canonical_shape({(x, 6) for x in range(3, 14)}),
        _canonical_shape({(5, 3), (6, 4), (7, 5), (8, 6), (9, 7), (10, 8)}),
    ]
    training = [
        _row(f"p{index}", vector, label=1) for index, vector in enumerate(positive_vectors, start=1)
    ] + [
        _row(f"n{index}", vector, label=0) for index, vector in enumerate(negative_vectors, start=1)
    ]
    shifted_positive = _row(
        "held-positive",
        _canonical_shape({(8, 6), (9, 6), (10, 6), (9, 5), (9, 7)}),
        label=1,
    )
    held_negative = _row(
        "held-negative",
        _canonical_shape({(4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6)}),
        label=0,
    )

    for scorer_kind in spike.SCORER_KINDS:
        scorer = spike._fit_patch_scorer(
            training,
            patch_id="binary_raw",
            scorer_kind=scorer_kind,
        )
        assert scorer.score(shifted_positive.candidate) > scorer.score(held_negative.candidate)


def _scaled_notehead_image(*, staff_spacing: int) -> Image.Image:
    size = staff_spacing * 4
    center = size // 2
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    radius_x = round(staff_spacing * 0.28)
    radius_y = round(staff_spacing * 0.20)
    draw.ellipse(
        (
            center - radius_x,
            center - radius_y,
            center + radius_x,
            center + radius_y,
        ),
        fill=0,
    )
    return image


def _binary_vector(width: int, height: int, active: set[tuple[int, int]]) -> tuple[float, ...]:
    return tuple(float((x, y) in active) for y in range(height) for x in range(width))


def _canonical_shape(active: set[tuple[int, int]]) -> tuple[float, ...]:
    return _binary_vector(spike.PATCH_WIDTH, spike.PATCH_HEIGHT, active)


def _row(candidate_id: str, vector: tuple[float, ...], *, label: int) -> spike.LabeledCandidate:
    candidate = spike.CandidatePatch(
        measure=1,
        id=candidate_id,
        rank=1,
        center_x=0.0,
        center_y=0.0,
        bbox=(0, 0, 1, 1),
        detector_score=0.0,
        patches={"binary_raw": vector},
    )
    return spike.LabeledCandidate(candidate=candidate, label=label)
