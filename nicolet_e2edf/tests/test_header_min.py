from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nicolet_e2edf.nicolet.header import (
    _infer_reference,
    _parse_derivation_fixed_record_montage_rows,
    _parse_unknown_montage_catalog_rows,
    _parse_supplemental_av_montage_rows,
    read_nervus_header,
)


def test_read_nervus_header_rejects_invalid_legacy_files(tmp_path: Path) -> None:
    """Legacy layouts should fail gracefully when file contents are invalid."""

    legacy = tmp_path / "legacy.eeg"
    legacy.write_bytes((0).to_bytes(4, "little") * 7)
    with pytest.raises(ValueError, match="Unsupported legacy Nicolet file format"):
        read_nervus_header(legacy)


def test_parse_supplemental_av_montage_rows_extracts_expected_rows() -> None:
    text = (
        "P10-av\x00"
        "23\x00"
        "AV26\x00"
        "\x01\x00"
        "CZ-PZ\x00"
        "26\x00"
        "27\x00"
        "0\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == [
        {
            "montageName": "AV26",
            "derivationName": "P10-av",
            "signalName1": "23",
            "signalName2": "AV26",
        }
    ]


def test_parse_supplemental_av_montage_rows_parses_numeric_derivation_pairs() -> None:
    text = (
        "CZ-PZ\x00"
        "26\x00"
        "27\x00"
        "1\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == [
        {
            "montageName": "",
            "derivationName": "CZ-PZ",
            "signalName1": "26",
            "signalName2": "27",
        }
    ]


def test_parse_supplemental_av_montage_rows_supports_shared_av_context() -> None:
    text = (
        "64 AV\x00"
        "Fp1 - av\x00"
        "1\x00"
        "Fp2 - av\x00"
        "2\x00"
        "AF3 - av\x00"
        "5\x00"
        "AV64\x00"
        "\x01\x00"
        "F2 -av\x00"
        "16\x00"
        "FT9 - av\x00"
        "17\x00"
        "AV64\x00"
        "\x01\x00"
        "EKG\x00"
        "68\x00"
        "\x01\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == [
        {
            "montageName": "AV64",
            "derivationName": "Fp1 - av",
            "signalName1": "1",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "Fp2 - av",
            "signalName1": "2",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "AF3 - av",
            "signalName1": "5",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "F2 -av",
            "signalName1": "16",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "FT9 - av",
            "signalName1": "17",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "EKG",
            "signalName1": "68",
            "signalName2": "",
        },
    ]


def test_parse_derivation_fixed_record_montage_rows_extracts_rows() -> None:
    stride = 520

    def _make_chunk(rows: list[tuple[str, str]]) -> bytes:
        chunk = bytearray(stride * (len(rows) + 1) + 64)

        def _put_utf16(rec_idx: int, offset: int, text: str) -> None:
            raw = text.encode("utf-16le")
            start = rec_idx * stride + offset
            chunk[start : start + len(raw)] = raw

        _put_utf16(0, 40, "MONTAGE128")
        for rec_idx, (name, signal_id) in enumerate(rows, start=1):
            _put_utf16(rec_idx, 232, name)
            _put_utf16(rec_idx, 360, signal_id)
        return bytes(chunk)

    # Too few rows should be ignored by the heuristic guard.
    small = _make_chunk([("F3", "69"), ("C3", "70"), ("CZ", "82")])
    assert _parse_derivation_fixed_record_montage_rows(small) == []

    rows = _parse_derivation_fixed_record_montage_rows(
        _make_chunk(
            [
                ("VTP1", "1"),
                ("VTP2", "2"),
                ("HST1", "23"),
                ("HST2", "24"),
                ("F3", "69"),
                ("C3", "70"),
                ("CZ", "82"),
                ("PZ", "83"),
            ]
        )
    )
    assert len(rows) == 8
    assert rows[0] == {
        "montageName": "MONTAGE128",
        "derivationName": "VTP1",
        "signalName1": "1",
        "signalName2": "",
        "source": "derivation_fixed_table",
    }
    assert rows[-1]["derivationName"] == "PZ"
    assert rows[-1]["signalName1"] == "83"


def test_parse_unknown_montage_catalog_rows_extracts_kanaler_table() -> None:
    entries = [
        ("FP1", "1"),
        ("FP2", "2"),
        ("AF7", "3"),
        ("AF8", "4"),
        ("F7", "5"),
        ("F8", "6"),
        ("F3", "7"),
        ("F4", "8"),
        ("T7", "9"),
        ("T8", "10"),
        ("C3", "11"),
        ("C4", "12"),
        ("P7", "13"),
        ("P8", "14"),
        ("O1", "15"),
        ("O2", "16"),
        ("CZ", "17"),
        ("PZ", "18"),
    ]
    text = "noise\x00misc\x00" + "32 kanaler\x00" + "".join(f"{name}\x00{idx}\x00" for name, idx in entries)
    rows = _parse_unknown_montage_catalog_rows(text.encode("utf-16le"))
    assert len(rows) == len(entries)
    assert rows[0] == {
        "montageName": "32 kanaler",
        "derivationName": "FP1",
        "signalName1": "1",
        "signalName2": "",
        "source": "unknown_montage_catalog",
    }
    assert rows[-1]["derivationName"] == "PZ"
    assert rows[-1]["signalName1"] == "18"


def test_parse_unknown_montage_catalog_rows_extracts_named_catalog() -> None:
    entries = [
        ("VTP1", "1"),
        ("VTP2", "2"),
        ("VTP3", "3"),
        ("VTP4", "4"),
        ("VTP5", "5"),
        ("VTP6", "6"),
        ("VST1", "7"),
        ("VST2", "8"),
        ("VST3", "9"),
        ("VST4", "10"),
        ("VST5", "11"),
        ("VST6", "12"),
        ("HTP1", "13"),
        ("HTP2", "14"),
        ("HST1", "15"),
        ("HST2", "16"),
        ("F3", "17"),
        ("C3", "18"),
        ("CZ", "19"),
        ("PZ", "20"),
        ("F4", "21"),
        ("C4", "22"),
        ("P4", "23"),
        ("EKG", "24"),
    ]
    text = "junk\x00\x00PRECATALOG\x00Q\x00Q\x00" + "".join(f"{name}\x00{idx}\x00" for name, idx in entries)
    rows = _parse_unknown_montage_catalog_rows(text.encode("utf-16le"))
    assert len(rows) == len(entries)
    assert rows[0]["montageName"] == "PRECATALOG"
    assert rows[0]["source"] == "unknown_montage_catalog"
    assert rows[0]["derivationName"] == "VTP1"
    assert rows[0]["signalName1"] == "1"


def test_parse_unknown_montage_catalog_rows_does_not_union_repeated_named_title_chunks() -> None:
    first_chunk = "".join(f"F{i}\x00{i}\x00" for i in range(1, 28))
    second_chunk = "".join(f"P{i}\x00{i+27}\x00" for i in range(1, 38))
    text = (
        "noise\x00"
        "Copy of As Recorded\x00"
        "\x1b\x00\x1b\x00"
        + first_chunk
        + ("junk\x00" * 40)
        + "Copy of As Recorded\x00"
        "\x1b\x00\x1b\x00"
        + second_chunk
    )
    rows = _parse_unknown_montage_catalog_rows(text.encode("utf-16le"))
    rows = [r for r in rows if r["montageName"] == "COPY OF AS RECORDED"]
    # Prefer the first chunk (starts at 1) instead of union-merging the later
    # repeated title occurrence that starts at a higher signal ID.
    assert len(rows) == 27
    assert rows[0]["derivationName"] == "F1"
    assert rows[-1]["signalName1"] == "27"


def test_infer_reference_treats_ref_case_variants_as_common() -> None:
    segments = [SimpleNamespace(refName=["Ref", "REF", "ref", ""])]
    assert _infer_reference(segments) == "common"
