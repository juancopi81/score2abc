from pathlib import Path

from PIL import Image, ImageDraw

from scripts.experiments import spike_notehead_affine_rectification as spike


def test_fit_and_rectify_parallel_staff_without_ground_truth(tmp_path: Path) -> None:
    width, height = 400, 100
    center_x = (width - 1) / 2
    expected_slope = 0.02
    flat_lines = [20, 30, 40, 50, 60]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for flat_y in flat_lines:
        draw.line(
            (
                0,
                flat_y + expected_slope * (0 - center_x),
                width - 1,
                flat_y + expected_slope * (width - 1 - center_x),
            ),
            fill="black",
            width=1,
        )
    draw.line((120, 12, 120, 75), fill="black", width=2)

    model = spike.fit_parallel_staff_model(image, flat_lines)

    assert 0.005 < model.slope <= expected_slope
    assert model.support > model.zero_slope_support
    assert all(
        abs(actual - expected) <= 1
        for actual, expected in zip(model.flat_lines, flat_lines, strict=True)
    )

    rectified = spike.rectify_measure_image(image, model=model, x_left_in_system=0)
    rectified.save(tmp_path / "rectified.png")
    source_x, source_y = spike.map_rectified_point_to_source(
        180,
        40,
        model=model,
        x_left_in_system=0,
    )
    assert source_x == 180
    assert source_y == 40 + model.slope * (180 - model.center_x)
