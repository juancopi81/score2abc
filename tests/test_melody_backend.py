from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from score2abc.melody import (
    INTERMEDIATE_MUSICXML_FILENAME,
    AudiverisMusicXMLBackend,
    FixtureMusicXMLBackend,
    HomrMusicXMLBackend,
    MusicXMLBackendError,
    build_musicxml_backend,
)
from score2abc.schemas import WorkItem, WorkMetadata

_TINY_MUSICXML = """\
<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>3</beats><beat-type>4</beat-type></time>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>1</duration>
      </note>
    </measure>
  </part>
</score-partwise>
"""


def _work_item(tmp_path: Path, slug: str = "demo-slug") -> WorkItem:
    pdf_path = tmp_path / f"{slug}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")
    return WorkItem(
        slug=slug,
        pdf_path=pdf_path,
        metadata=WorkMetadata(
            title="Demo",
            composer="Composer",
            rhythm="Pasillo",
            time_signature="3/4",
            key_hint="Em",
        ),
    )


def _write_musicxml(path: Path, contents: str = _TINY_MUSICXML) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")
    return path


def _write_image(path: Path, *, size: tuple[int, int] = (24, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)
    return path


def _write_fake_homr(path: Path, contents: str = _TINY_MUSICXML) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""\
#!/bin/sh
set -eu
input="$1"
test -f "$input"
output="${{input%.*}}.musicxml"
cat > "$output" <<'XML'
{textwrap.dedent(contents).lstrip()}XML
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_fake_audiveris(
    path: Path,
    *,
    contents: str = _TINY_MUSICXML,
    output_extension: str = ".xml",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""\
#!{sys.executable}
import pathlib
import sys
import zipfile

args = sys.argv[1:]
for required in ("-batch", "-transcribe", "-export", "-constant", "-output", "--"):
    assert required in args, args
constant_value = args[args.index("-constant") + 1]
assert constant_value == "org.audiveris.omr.sheet.BookManager.useCompression=false"
output_dir = pathlib.Path(args[args.index("-output") + 1])
input_path = pathlib.Path(args[-1])
assert input_path.is_absolute()
assert input_path.exists()
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / (input_path.stem + "{output_extension}")
payload = {textwrap.dedent(contents).lstrip()!r}
if "{output_extension}" == ".mxl":
    with zipfile.ZipFile(output_path, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("score.xml", payload)
else:
    output_path.write_text(payload, encoding="utf-8")
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_fixture_backend_copies_and_validates_musicxml(tmp_path: Path) -> None:
    source_dir = tmp_path / "musicxml"
    item = _work_item(tmp_path)
    fixture = _write_musicxml(source_dir / f"{item.slug}.musicxml")
    work_dir = tmp_path / "out" / item.slug

    backend = FixtureMusicXMLBackend(source_dir=source_dir)
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.source_path == fixture
    expected_output = work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME
    assert result.output_path == expected_output
    assert expected_output.exists()
    assert expected_output.read_text(encoding="utf-8") == fixture.read_text(encoding="utf-8")


def test_fixture_backend_returns_none_when_no_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "musicxml"
    source_dir.mkdir()
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug

    backend = FixtureMusicXMLBackend(source_dir=source_dir)
    assert backend.produce_musicxml(item=item, work_dir=work_dir) is None
    assert not (work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME).exists()


def test_fixture_backend_accepts_xml_extension(tmp_path: Path) -> None:
    source_dir = tmp_path / "musicxml"
    item = _work_item(tmp_path)
    _write_musicxml(source_dir / f"{item.slug}.xml")
    work_dir = tmp_path / "out" / item.slug

    backend = FixtureMusicXMLBackend(source_dir=source_dir)
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.source_path.suffix == ".xml"


def test_fixture_backend_raises_when_source_is_invalid(tmp_path: Path) -> None:
    source_dir = tmp_path / "musicxml"
    item = _work_item(tmp_path)
    bad_fixture = source_dir / f"{item.slug}.musicxml"
    bad_fixture.parent.mkdir(parents=True)
    bad_fixture.write_text("not really xml", encoding="utf-8")
    work_dir = tmp_path / "out" / item.slug

    backend = FixtureMusicXMLBackend(source_dir=source_dir)

    with pytest.raises(MusicXMLBackendError):
        backend.produce_musicxml(item=item, work_dir=work_dir)

    # Validation failure must not leave a partial intermediate file behind.
    assert not (work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME).exists()


def test_fixture_backend_raises_when_musicxml_has_no_time_signature(tmp_path: Path) -> None:
    source_dir = tmp_path / "musicxml"
    item = _work_item(tmp_path)
    _write_musicxml(
        source_dir / f"{item.slug}.musicxml",
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1"><part-name>Music</part-name></score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <note>
                <pitch><step>C</step><octave>4</octave></pitch>
                <duration>1</duration>
              </note>
            </measure>
          </part>
        </score-partwise>
        """,
    )
    work_dir = tmp_path / "out" / item.slug

    backend = FixtureMusicXMLBackend(source_dir=source_dir)

    with pytest.raises(MusicXMLBackendError, match="time signature"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_build_musicxml_backend_returns_fixture_backend(tmp_path: Path) -> None:
    backend = build_musicxml_backend(source_dir=tmp_path / "musicxml")
    assert isinstance(backend, FixtureMusicXMLBackend)
    assert backend.name == "fixture"


def test_build_musicxml_backend_returns_homr_backend(tmp_path: Path) -> None:
    backend = build_musicxml_backend(
        source_dir=tmp_path / "musicxml",
        backend="homr",
        homr_command="/usr/bin/false",
        homr_input="deskewed-page",
    )
    assert isinstance(backend, HomrMusicXMLBackend)
    assert backend.name == "homr"


def test_build_musicxml_backend_returns_audiveris_backend(tmp_path: Path) -> None:
    backend = build_musicxml_backend(
        source_dir=tmp_path / "musicxml",
        backend="audiveris",
        audiveris_command="/usr/bin/false",
        audiveris_input="deskewed-page",
    )
    assert isinstance(backend, AudiverisMusicXMLBackend)
    assert backend.name == "audiveris"


def test_build_musicxml_backend_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported MusicXML backend"):
        build_musicxml_backend(source_dir=tmp_path / "musicxml", backend="demo")


def test_build_musicxml_backend_rejects_unknown_homr_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported homr input mode"):
        build_musicxml_backend(
            source_dir=tmp_path / "musicxml",
            backend="homr",
            homr_input="demo",
        )


def test_build_musicxml_backend_rejects_unknown_audiveris_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported audiveris input mode"):
        build_musicxml_backend(
            source_dir=tmp_path / "musicxml",
            backend="audiveris",
            audiveris_input="demo",
        )


def test_homr_backend_runs_external_command_and_validates_musicxml(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr))
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.output_path == work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME
    assert result.output_path.exists()
    assert result.source_path == work_dir / "intermediate" / "homr" / "page_001.png"


def test_homr_backend_uses_deskewed_page_input(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    deskewed_path = work_dir / "systems" / "page_001_deskewed.png"
    deskewed_path.parent.mkdir(parents=True)
    deskewed_path.write_bytes(b"fake png")
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr), input_mode="deskewed-page")
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.output_path.exists()
    assert result.source_path == work_dir / "intermediate" / "homr" / "page_001_deskewed.png"


def test_homr_backend_creates_systems_collage_input(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    _write_image(work_dir / "systems" / "system_001.png", size=(80, 20))
    _write_image(work_dir / "systems" / "system_002.png", size=(120, 30))
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr), input_mode="systems")
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    collage_path = work_dir / "intermediate" / "homr" / "systems_collage.png"
    assert result is not None
    assert result.source_path == collage_path
    assert result.output_path.exists()
    with Image.open(collage_path) as collage:
        assert collage.size == (120, 74)


def test_homr_backend_passes_absolute_input_when_work_dir_is_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _work_item(tmp_path)
    monkeypatch.chdir(tmp_path)
    work_dir = Path("out") / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr))
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.output_path == work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME
    assert result.output_path.exists()


def test_homr_backend_raises_when_command_is_missing(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")

    backend = HomrMusicXMLBackend(command=str(tmp_path / "missing-homr"))

    with pytest.raises(MusicXMLBackendError, match="homr command not found"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_homr_backend_raises_when_no_rendered_pages(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr))

    with pytest.raises(MusicXMLBackendError, match="No rendered page images"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_homr_backend_raises_when_no_deskewed_pages(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr), input_mode="deskewed-page")

    with pytest.raises(MusicXMLBackendError, match="No deskewed page images"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_homr_backend_raises_when_no_system_crops(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr), input_mode="systems")

    with pytest.raises(MusicXMLBackendError, match="No system crops"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_homr_backend_rejects_multiple_pages_for_now(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    pages_dir = work_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "page_001.png").write_bytes(b"fake png")
    (pages_dir / "page_002.png").write_bytes(b"fake png")
    fake_homr = _write_fake_homr(tmp_path / "bin" / "homr")

    backend = HomrMusicXMLBackend(command=str(fake_homr))

    with pytest.raises(MusicXMLBackendError, match="supports one rendered page"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_cli_rejects_unknown_homr_input() -> None:
    from score2abc.cli import build_parser

    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "out", "--homr-input", "demo"])


def test_audiveris_backend_runs_external_command_and_validates_musicxml(
    tmp_path: Path,
) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")
    fake_audiveris = _write_fake_audiveris(tmp_path / "bin" / "audiveris")

    backend = AudiverisMusicXMLBackend(command=str(fake_audiveris))
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.output_path == work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME
    assert result.output_path.exists()
    assert result.source_path == work_dir / "intermediate" / "audiveris" / "page_001.png"
    assert result.raw_output_path == result.source_path.with_suffix(".xml")


def test_audiveris_backend_uses_deskewed_page_input(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    deskewed_path = work_dir / "systems" / "page_001_deskewed.png"
    deskewed_path.parent.mkdir(parents=True)
    deskewed_path.write_bytes(b"fake png")
    fake_audiveris = _write_fake_audiveris(tmp_path / "bin" / "audiveris")

    backend = AudiverisMusicXMLBackend(
        command=str(fake_audiveris),
        input_mode="deskewed-page",
    )
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.output_path.exists()
    assert result.source_path == work_dir / "intermediate" / "audiveris" / "page_001_deskewed.png"


def test_audiveris_backend_creates_systems_collage_input(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    _write_image(work_dir / "systems" / "system_001.png", size=(80, 20))
    _write_image(work_dir / "systems" / "system_002.png", size=(120, 30))
    fake_audiveris = _write_fake_audiveris(tmp_path / "bin" / "audiveris")

    backend = AudiverisMusicXMLBackend(command=str(fake_audiveris), input_mode="systems")
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    collage_path = work_dir / "intermediate" / "audiveris" / "systems_collage.png"
    assert result is not None
    assert result.source_path == collage_path
    assert result.output_path.exists()
    with Image.open(collage_path) as collage:
        assert collage.size == (120, 74)


def test_audiveris_backend_extracts_mxl_output(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")
    fake_audiveris = _write_fake_audiveris(
        tmp_path / "bin" / "audiveris",
        output_extension=".mxl",
    )

    backend = AudiverisMusicXMLBackend(command=str(fake_audiveris))
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert result.raw_output_path == result.source_path.with_suffix(".mxl")
    assert result.output_path.read_text(encoding="utf-8").startswith("<?xml")


def test_audiveris_backend_raises_when_command_is_missing(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")

    backend = AudiverisMusicXMLBackend(command=str(tmp_path / "missing-audiveris"))

    with pytest.raises(MusicXMLBackendError, match="audiveris command not found"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_audiveris_backend_raises_when_no_rendered_pages(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    fake_audiveris = _write_fake_audiveris(tmp_path / "bin" / "audiveris")

    backend = AudiverisMusicXMLBackend(command=str(fake_audiveris))

    with pytest.raises(MusicXMLBackendError, match="No rendered page images"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_audiveris_backend_raises_when_command_fails(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")
    failing_audiveris = tmp_path / "bin" / "audiveris"
    failing_audiveris.parent.mkdir(parents=True)
    failing_audiveris.write_text("#!/bin/sh\necho failed >&2\nexit 2\n", encoding="utf-8")
    failing_audiveris.chmod(0o755)

    backend = AudiverisMusicXMLBackend(command=str(failing_audiveris))

    with pytest.raises(MusicXMLBackendError, match="audiveris failed"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_audiveris_backend_raises_when_no_output(tmp_path: Path) -> None:
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    page_path = work_dir / "pages" / "page_001.png"
    page_path.parent.mkdir(parents=True)
    page_path.write_bytes(b"fake png")
    noop_audiveris = tmp_path / "bin" / "audiveris"
    noop_audiveris.parent.mkdir(parents=True)
    noop_audiveris.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    noop_audiveris.chmod(0o755)

    backend = AudiverisMusicXMLBackend(command=str(noop_audiveris))

    with pytest.raises(MusicXMLBackendError, match="did not produce"):
        backend.produce_musicxml(item=item, work_dir=work_dir)


def test_cli_accepts_audiveris_backend() -> None:
    from score2abc.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "out",
            "--musicxml-backend",
            "audiveris",
            "--audiveris-input",
            "systems",
        ]
    )

    assert args.musicxml_backend == "audiveris"
    assert args.audiveris_input == "systems"


def test_cli_rejects_unknown_audiveris_input() -> None:
    from score2abc.cli import build_parser

    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "out", "--audiveris-input", "demo"])


def test_backend_output_is_picked_up_by_find_musicxml_source(tmp_path: Path) -> None:
    """Backend writes to the same path the pipeline reads from in extract_melody."""
    from score2abc.pipeline import _find_musicxml_source

    source_dir = tmp_path / "musicxml"
    item = _work_item(tmp_path)
    _write_musicxml(source_dir / f"{item.slug}.musicxml")
    work_dir = tmp_path / "out" / item.slug

    backend = build_musicxml_backend(source_dir=source_dir)
    result = backend.produce_musicxml(item=item, work_dir=work_dir)

    assert result is not None
    assert _find_musicxml_source(work_dir) == result.output_path


def test_manual_drop_is_preserved_when_no_fixture(tmp_path: Path) -> None:
    """If a user manually placed intermediate/musicxml.xml, no-source skip leaves it alone."""
    from score2abc.pipeline import _find_musicxml_source

    source_dir = tmp_path / "musicxml"
    source_dir.mkdir()
    item = _work_item(tmp_path)
    work_dir = tmp_path / "out" / item.slug
    intermediate = work_dir / "intermediate"
    intermediate.mkdir(parents=True)
    manual_path = _write_musicxml(intermediate / INTERMEDIATE_MUSICXML_FILENAME)

    backend = build_musicxml_backend(source_dir=source_dir)
    assert backend.produce_musicxml(item=item, work_dir=work_dir) is None
    assert manual_path.exists()
    assert _find_musicxml_source(work_dir) == manual_path
