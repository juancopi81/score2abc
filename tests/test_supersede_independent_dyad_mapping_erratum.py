import pytest

from scripts.experiments import supersede_independent_dyad_mapping_erratum as erratum


def test_accepts_only_the_audited_no_lo_creas_mapping_correction() -> None:
    prior = {
        "automatic_crops": [
            {
                "automatic_crop_index": 1,
                "physical_measure_spans": [{"measure_number": 1, "note_start": 0, "note_end": 9}],
            },
            {
                "automatic_crop_index": 2,
                "physical_measure_spans": [{"measure_number": 1, "note_start": 9, "note_end": 12}],
            },
        ]
    }
    corrected = {
        "automatic_crops": [
            {
                "automatic_crop_index": 1,
                "physical_measure_spans": [{"measure_number": 1, "note_start": 0, "note_end": 6}],
            },
            {
                "automatic_crop_index": 2,
                "physical_measure_spans": [{"measure_number": 1, "note_start": 6, "note_end": 12}],
            },
        ]
    }

    changes = erratum._mapping_diff(prior, corrected)

    erratum._validate_narrow_no_lo_creas_correction(changes)
    assert [change["automatic_crop_index"] for change in changes] == [1, 2]


def test_rejects_any_additional_mapping_change() -> None:
    changes = [
        {
            "automatic_crop_index": 1,
            "before": [{"measure_number": 1, "note_start": 0, "note_end": 9}],
            "after": [{"measure_number": 1, "note_start": 0, "note_end": 6}],
        },
        {
            "automatic_crop_index": 2,
            "before": [{"measure_number": 1, "note_start": 9, "note_end": 12}],
            "after": [{"measure_number": 1, "note_start": 6, "note_end": 12}],
        },
        {
            "automatic_crop_index": 3,
            "before": [{"measure_number": 2, "note_start": 0, "note_end": 3}],
            "after": [{"measure_number": 2, "note_start": 0, "note_end": 2}],
        },
    ]

    with pytest.raises(ValueError, match="only No lo Creas crops 1 and 2"):
        erratum._validate_narrow_no_lo_creas_correction(changes)
