# score2abc

Pipeline to convert handwritten Colombian scores into ABC notation plus metadata.

## Dataset

`dataset/` is the current golden dataset of source PDFs, with a canonical metadata
table at `dataset/metadata.csv`. These files come from the
manuscript transcriptions of Colombian organist Jaime Llanos Gonzalez, later
photocopied and shared by triplista Jairo Rincon Gomez (June 2002) with the music
library of Universidad de Antioquia. The photocopies are low-legibility and in
fragile condition, so this dataset prioritizes preservation and accurate transcription.

## Filename convention (when possible)

`jaime-llanos_<num>_<titulo>_<genero>_<autor>.pdf`

Normalization rules:
- lowercase
- ASCII only (accents removed)
- words separated by hyphens
- punctuation removed (e.g., "Gato'e" -> `gatoe`)
- author last names separated by hyphens

Not all future sources will have clean or consistent names. If a file cannot be
normalized cleanly, keep the original filename and rely on the metadata table
as the source of truth for title/composer/genre.
