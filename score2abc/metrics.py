from __future__ import annotations

from typing import Dict


def compare_events(pred: Dict, truth: Dict) -> Dict[str, object]:
    pred_notes = pred.get("notes") or []
    truth_notes = truth.get("notes") or []
    pred_chords = pred.get("chords") or []
    truth_chords = truth.get("chords") or []

    pred_ts = pred.get("time_signature")
    truth_ts = truth.get("time_signature")
    time_signature_match = (
        pred_ts is not None and truth_ts is not None and pred_ts == truth_ts
    )

    return {
        "pred_notes": len(pred_notes),
        "truth_notes": len(truth_notes),
        "note_count_delta": len(pred_notes) - len(truth_notes),
        "note_count_match": len(pred_notes) == len(truth_notes),
        "pred_chords": len(pred_chords),
        "truth_chords": len(truth_chords),
        "chord_count_delta": len(pred_chords) - len(truth_chords),
        "chord_count_match": len(pred_chords) == len(truth_chords),
        "time_signature_pred": pred_ts,
        "time_signature_truth": truth_ts,
        "time_signature_match": time_signature_match,
    }
