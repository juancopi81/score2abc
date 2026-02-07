from __future__ import annotations

from typing import Dict, Iterable, Tuple


def compare_events(pred: Dict, truth: Dict) -> Dict[str, object]:
    pred_notes = pred.get("notes") or []
    truth_notes = truth.get("notes") or []
    pred_chords = pred.get("chords") or []
    truth_chords = truth.get("chords") or []

    pred_ts = pred.get("time_signature")
    truth_ts = truth.get("time_signature")
    time_signature_match = pred_ts is not None and truth_ts is not None and pred_ts == truth_ts

    pred_note_set = _normalize_notes(pred_notes)
    truth_note_set = _normalize_notes(truth_notes)
    pred_chord_set = _normalize_chords(pred_chords)
    truth_chord_set = _normalize_chords(truth_chords)

    note_stats = _precision_recall_f1(pred_note_set, truth_note_set)
    chord_stats = _precision_recall_f1(pred_chord_set, truth_chord_set)

    return {
        "pred_notes": len(pred_notes),
        "truth_notes": len(truth_notes),
        "note_count_delta": len(pred_notes) - len(truth_notes),
        "note_count_match": len(pred_notes) == len(truth_notes),
        "note_true_positives": note_stats["tp"],
        "note_false_positives": note_stats["fp"],
        "note_false_negatives": note_stats["fn"],
        "note_precision": note_stats["precision"],
        "note_recall": note_stats["recall"],
        "note_f1": note_stats["f1"],
        "pred_chords": len(pred_chords),
        "truth_chords": len(truth_chords),
        "chord_count_delta": len(pred_chords) - len(truth_chords),
        "chord_count_match": len(pred_chords) == len(truth_chords),
        "chord_true_positives": chord_stats["tp"],
        "chord_false_positives": chord_stats["fp"],
        "chord_false_negatives": chord_stats["fn"],
        "chord_precision": chord_stats["precision"],
        "chord_recall": chord_stats["recall"],
        "chord_f1": chord_stats["f1"],
        "time_signature_pred": pred_ts,
        "time_signature_truth": truth_ts,
        "time_signature_match": time_signature_match,
    }


def _normalize_notes(notes: Iterable[Dict]) -> set[Tuple[object, ...]]:
    normalized: set[Tuple[object, ...]] = set()
    for note in notes:
        normalized.add(
            (
                int(note.get("measure", 0)),
                _round_float(note.get("onset_beats", 0.0)),
                _round_float(note.get("duration_beats", 0.0)),
                int(note.get("pitch_midi", 0)),
            )
        )
    return normalized


def _normalize_chords(chords: Iterable[Dict]) -> set[Tuple[object, ...]]:
    normalized: set[Tuple[object, ...]] = set()
    for chord in chords:
        normalized.add(
            (
                int(chord.get("measure", 0)),
                _round_float(chord.get("onset_beats", 0.0)),
                str(chord.get("symbol", "")).strip(),
            )
        )
    return normalized


def _precision_recall_f1(
    pred: set[Tuple[object, ...]], truth: set[Tuple[object, ...]]
) -> Dict[str, float | int]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _round_float(value: object) -> float:
    return round(float(value), 4)
