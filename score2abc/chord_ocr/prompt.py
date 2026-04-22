from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a chord-symbol OCR assistant for handwritten Latin American scores "
    "(pasillos, bambucos, danzas). The input is a cropped strip above or below a "
    "single staff system. Return only chord symbols such as 'Em', 'B7', 'D/F#', "
    "or 'Cmaj7'. Do NOT return lyrics, dynamics, tempo or expression marks, "
    "rehearsal letters, measure numbers, or standalone numbers. Preserve the "
    "left-to-right reading order. For each symbol, report its approximate "
    "horizontal position as a fraction of the crop width in [0, 1], and a "
    "confidence in [0, 1]. Respond with strict JSON matching the given schema."
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "detections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "symbol": {"type": "STRING"},
                    "x_fraction": {"type": "NUMBER"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["symbol", "x_fraction", "confidence"],
            },
        }
    },
    "required": ["detections"],
}
