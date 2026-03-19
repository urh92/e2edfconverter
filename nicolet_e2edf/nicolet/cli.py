from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import re
from fractions import Fraction
from datetime import datetime, timedelta
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .data import read_nervus_data
from .edf_writer import write_edf
from .header import canonical_event_text, normalize_events, read_nervus_header
from .montage_recovery import (
    DIRECT_ID_NAME_SOURCES,
    build_montage_recovery_plan,
    should_skip_main_direct_numeric_ref_fallback,
)
from .types import EventItem, NervusHeader, SegmentInfo, SleepScoreInfo
from .tui import TuiOptions, rich_available, run_rich_wizard, run_tui

logger = logging.getLogger(__name__)
_SFREQ_INT_TOL = 1e-6
_EEG_POSITION_PREFIXES = {
    "FP",
    "AF",
    "F",
    "FC",
    "FT",
    "C",
    "CP",
    "TP",
    "T",
    "P",
    "PO",
    "O",
    "I",
}
_NUMERIC_REF_LABEL_RE = re.compile(r"^\d+\s*-\s*REF$", re.IGNORECASE)
_SSE_64_NUMERIC_ID_EEG_LABELS = [
    "FP1",
    "FP2",
    "AF7",
    "AF8",
    "AF3",
    "AF4",
    "F9",
    "F10",
    "F7",
    "F8",
    "F5",
    "F6",
    "F3",
    "F4",
    "F1",
    "F2",
    "FT9",
    "FT10",
    "FT7",
    "FT8",
    "FC3",
    "FC4",
    "T9",
    "T10",
    "T7",
    "T8",
    "C5",
    "C6",
    "C3",
    "C4",
    "C1",
    "C2",
    "TP9",
    "TP10",
    "TP7",
    "TP8",
    "CP3",
    "CP4",
    "P9",
    "P10",
    "P7",
    "P8",
    "P5",
    "P6",
    "P3",
    "P4",
    "P1",
    "P2",
    "PO7",
    "PO8",
    "PO3",
    "PO4",
    "O1",
    "O2",
    "FPZ",
    "AFZ",
    "FZ",
    "FCZ",
    "CZ",
    "CPZ",
    "PZ",
    "POZ",
    "OZ",
]
_SSE_64_NUMERIC_ID_MAP = {
    str(index): label for index, label in enumerate(_SSE_64_NUMERIC_ID_EEG_LABELS, start=1)
}
_SSE_64_NUMERIC_ID_MAP["64"] = "EKG"


def _parse_integer_hz(raw: str) -> float:
    """argparse type parser that accepts only positive integer Hz values."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid sampling rate: {raw}") from exc
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Sampling rate must be a positive finite value")
    rounded = int(round(value))
    if rounded <= 0 or not np.isclose(value, float(rounded), rtol=0.0, atol=_SFREQ_INT_TOL):
        raise argparse.ArgumentTypeError(
            "Sampling rate for EDF output must be an integer Hz value (e.g. 128, 256, 512)"
        )
    return float(rounded)


def _require_integer_output_rate(fs: float, *, context: str) -> None:
    rounded = int(round(float(fs)))
    if rounded > 0 and np.isclose(float(fs), float(rounded), rtol=0.0, atol=_SFREQ_INT_TOL):
        return
    raise RuntimeError(
        f"{context} sampling rate is {fs} Hz, but EDF output currently requires integer Hz. "
        "Use --resample-to <integer Hz>."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nicolet-e2edf",
        description="Convert Nicolet/Nervus .e/.eeg EEG recordings into EDF files.",
    )
    parser.add_argument(
        "--in",
        dest="input_paths",
        nargs="*",
        help="Input .e file(s) and/or folder(s)",
    )
    parser.add_argument("--out", dest="output_dir", help="Output directory")
    parser.add_argument(
        "--glob",
        default="*.e",
        help="Glob pattern when input is a folder (default also picks up .eeg files)",
    )
    parser.add_argument(
        "--patient-json",
        type=Path,
        help="Optional JSON file with patient metadata glob rules",
    )
    parser.add_argument(
        "--json-sidecar",
        action="store_true",
        help="Write a JSON sidecar with channel metadata and events",
    )
    parser.add_argument(
        "--split-by-segment",
        action="store_true",
        help="Write one EDF per segment when recordings contain multiple segments",
    )
    parser.add_argument(
        "--vendor-style",
        action="store_true",
        help="Suppress system events to better match vendor EDF exports",
    )
    parser.add_argument(
        "--resample-to",
        type=_parse_integer_hz,
        help="Optional integer sampling rate (Hz) for EDF output (requires scipy)",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=None,
        help="Optional high-pass filter cutoff in Hz (e.g. 0.5)",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=None,
        help="Optional low-pass filter cutoff in Hz (e.g. 35)",
    )
    parser.add_argument(
        "--notch",
        type=float,
        default=None,
        help="Optional notch filter frequency in Hz (e.g. 50 or 60 for powerline)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--ui",
        dest="ui",
        action="store_true",
        help="Interactive terminal UI (wizard + animated progress; requires 'rich')",
    )
    # Backwards-compatible aliases (hidden): use --ui instead.
    parser.add_argument("--gui", dest="ui", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wiz", dest="ui", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tui", dest="ui", action="store_true", help=argparse.SUPPRESS)
    return parser


def _patient_meta_from_header(nrv_header) -> dict[str, str]:
    """Extract patient metadata from the parsed .e file header."""
    info = nrv_header.PatientInfo or {}

    first = str(info.get("firstName") or "").strip()
    last = str(info.get("lastName") or "").strip()
    name_parts = [p for p in [first, last] if p]
    patient_name = " ".join(name_parts) if name_parts else "X"

    # altID holds the CPR number (DDMMYY-XXXX); prefer it over the internal GUID in patientID
    patient_id = str(info.get("altID") or info.get("patientID") or "").strip() or "X"

    # sexID is stored as a Nicolet GUID; map known values to EDF M/F codes
    _SEX_GUID_MAP = {
        "{0ADD99E3-ACCA-11CF-9B9A-0800099E03CD}": "M",
        "{0ADD99E4-ACCA-11CF-9B9A-0800099E03CD}": "F",
    }
    sex_raw = str(info.get("sexID") or "").strip().upper()
    patient_sex = _SEX_GUID_MAP.get(sex_raw, sex_raw if sex_raw else "X")

    dob = info.get("DOB")
    if dob is not None and hasattr(dob, "strftime"):
        birthdate = dob.strftime("%Y-%m-%d")
    else:
        birthdate = str(dob).strip() if dob else ""

    meta: dict[str, str] = {
        "PatientName": patient_name,
        "PatientID": patient_id,
        "PatientSex": patient_sex,
        "StudyDescription": "Nicolet EEG export",
        "SeriesDescription": "Nicolet EEG export",
    }
    if birthdate:
        meta["PatientBirthDate"] = birthdate
    return meta


def _load_patient_rules(path: Path | None) -> list[dict[str, str]]:
    if not path:
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - handled in CLI execution
        raise SystemExit(f"Unable to parse patient JSON: {exc}") from exc
    if isinstance(data, dict):
        data = [data]
    rules: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict) or "glob" not in entry:
            raise SystemExit("Each patient rule must be an object with a 'glob' key")
        rules.append(entry)
    return rules


def _select_patient_metadata(
    path: Path, rules: Iterable[dict[str, str]], nrv_header=None
) -> dict[str, str]:
    base = _patient_meta_from_header(nrv_header) if nrv_header is not None else {}
    for rule in rules:
        pattern = rule.get("glob")
        if not isinstance(pattern, str):
            continue
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
            override = {k: v for k, v in rule.items() if k != "glob"}
            base.update(override)
            break
    return base


def _discover_inputs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        candidates: set[Path] = set()

        def _add(pattern: str) -> None:
            candidates.update(input_path.glob(pattern))

        normalized = glob_pattern.lower()
        if normalized.endswith(".e"):
            _add(glob_pattern)
            _add(glob_pattern[:-2] + ".eeg")
        elif normalized.endswith(".eeg"):
            _add(glob_pattern)
            _add(glob_pattern[:-4] + ".e")
        else:
            # Treat the glob as a name/path filter, and always target .e/.eeg.
            _add(f"{glob_pattern}.e")
            _add(f"{glob_pattern}.eeg")

        return sorted(candidates)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _discover_inputs_with_roots(
    input_paths: Iterable[Path],
    glob_pattern: str,
) -> list[tuple[Path, Path | None]]:
    discovered: list[tuple[Path, Path | None]] = []
    for input_path in input_paths:
        if input_path.is_file():
            discovered.append((input_path, None))
            continue
        if input_path.is_dir():
            for match in _discover_inputs(input_path, glob_pattern):
                discovered.append((match, input_path))
            continue
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return discovered


def _candidate_output_path(input_path: Path, output_dir: Path, input_root: Path | None) -> Path:
    if input_root and input_root.is_dir():
        try:
            relative = input_path.relative_to(input_root)
        except ValueError:
            relative = Path(input_path.name)
        return output_dir / relative.with_suffix(".edf")
    return output_dir / f"{input_path.stem}.edf"


def _clean_channel_label(label: object) -> str:
    cleaned = str(label or "").split("\x00", 1)[0].strip()
    return cleaned


def _channel_match_key(label: object) -> str:
    return _clean_channel_label(label).upper().replace(" ", "").replace("_", "")


def _normalize_eeg_position(label: object) -> str | None:
    token = _clean_channel_label(label).upper()
    if not token:
        return None

    token = token.replace("EEG", "", 1) if token.startswith("EEG ") else token
    token = token.replace(" ", "").replace("_", "")
    if token.endswith("-REF"):
        token = token[:-4]
    elif token.endswith("REF") and len(token) > 3:
        token = token[:-3]
    token = token.strip("-")

    if token in {"A1", "A2", "REF", "REFERENCE", "GROUND", "GND"}:
        return None
    if token == "IZ":
        return token

    if token.endswith("Z"):
        prefix = token[:-1]
        if prefix in _EEG_POSITION_PREFIXES:
            return token
        return None

    match = re.fullmatch(r"([A-Z]{1,3})(\d{1,2})", token)
    if not match:
        return None
    prefix, number_text = match.groups()
    if prefix not in _EEG_POSITION_PREFIXES:
        return None
    number = int(number_text)
    if 1 <= number <= 12:
        return f"{prefix}{number}"
    return None


def _extract_eeg_label_from_derivation(derivation_name: object) -> str | None:
    raw = _clean_channel_label(derivation_name)
    if not raw:
        return None
    lhs = re.split(r"[-/\\+,:;]", raw, maxsplit=1)[0]

    for part in (lhs, raw):
        normalized = _normalize_eeg_position(part)
        if normalized:
            return normalized
        for token in re.findall(r"[A-Za-z]{1,4}\s*\d{0,2}[A-Za-z]?", part):
            normalized = _normalize_eeg_position(token)
            if normalized:
                return normalized
    return None


def _normalize_derivation_endpoint(token: str) -> str | None:
    cleaned = _clean_channel_label(token).upper().replace(" ", "").replace("_", "")
    if not cleaned:
        return None
    # Derivation labels may include legacy aliases in parentheses, e.g. T7(T3).
    cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()
    if not cleaned:
        return None
    # Legacy derivation strings sometimes use 01/02 for occipitals.
    if cleaned == "01":
        return "O1"
    if cleaned == "02":
        return "O2"
    return _normalize_eeg_position(cleaned)


def _extract_eeg_endpoints_from_derivation(derivation_name: object) -> tuple[str | None, str | None]:
    raw = _clean_channel_label(derivation_name)
    if not raw:
        return None, None
    if "-" not in raw:
        return _extract_eeg_label_from_derivation(raw), None
    left_raw, right_raw = raw.split("-", 1)
    left = _normalize_derivation_endpoint(left_raw)
    right = _normalize_derivation_endpoint(right_raw)
    return left, right


def _extract_non_eeg_single_derivation_label(derivation_name: object) -> str | None:
    raw = _clean_channel_label(derivation_name)
    if not raw or "-" in raw:
        return None
    candidate = raw.upper()
    if _categorize_channel(candidate) in {"EKG", "EOG", "Reference", "Stimulus"}:
        return candidate
    return None


def _add_mapping_with_conflict_tracking(
    mapping: dict[str, str],
    conflicts: set[str],
    signal_name: str,
    electrode_name: str | None,
) -> None:
    if not signal_name or not electrode_name:
        return
    key = _channel_match_key(signal_name)
    previous = mapping.get(key)
    if previous and previous != electrode_name:
        conflicts.add(key)
        return
    mapping[key] = electrode_name


def _is_numeric_style_channel_label(label: str) -> bool:
    cleaned = _clean_channel_label(label)
    if not cleaned:
        return False
    return bool(re.fullmatch(r"[0-9]+", cleaned) or _NUMERIC_REF_LABEL_RE.fullmatch(cleaned))


def _is_sse_source_recording(header: object) -> bool:
    filename = getattr(header, "filename", None)
    if not filename:
        return False
    path_text = str(filename).replace("\\", "/").upper()
    return "/SSE/" in path_text or "SSE-ARKIV" in path_text


def _extract_numeric_placeholder_id(label: object) -> str | None:
    cleaned = _clean_channel_label(label)
    if not cleaned:
        return None
    if cleaned.isdigit():
        return str(int(cleaned))
    match = _NUMERIC_REF_LABEL_RE.fullmatch(cleaned.upper())
    if not match:
        return None
    return str(int(cleaned.split("-", 1)[0].strip()))


def _recover_sse_64_numeric_id_labels(header: object, channel_labels: list[str]) -> list[str] | None:
    """Apply a fixed SSE 64-contact numeric-ID map (63 EEG + channel ID 64 = EKG).

    This is intentionally strict and only triggers for in-house SSE recordings
    whose channel labels are an exact permutation of placeholder IDs 1..64
    (optionally in ``N-Ref`` form).
    """
    if len(channel_labels) != 64:
        return None
    if not _is_sse_source_recording(header):
        return None

    ids: list[str] = []
    for label in channel_labels:
        numeric_id = _extract_numeric_placeholder_id(label)
        if numeric_id is None:
            return None
        if numeric_id not in _SSE_64_NUMERIC_ID_MAP:
            return None
        ids.append(numeric_id)

    expected_ids = {str(i) for i in range(1, 65)}
    if set(ids) != expected_ids or len(set(ids)) != 64:
        return None

    recovered = [_SSE_64_NUMERIC_ID_MAP[numeric_id] for numeric_id in ids]
    logger.info("Applied SSE fixed 64-channel numeric-ID mapping (63 EEG + EKG)")
    return recovered


def _should_attempt_numeric_montage_recovery(channel_labels: list[str]) -> bool:
    """Return True when montage-based relabeling is likely to be beneficial.

    The recovery step only replaces placeholder labels (numeric / ``N-Ref``),
    so this gate is intentionally permissive when placeholders dominate even if
    a small number of named non-EEG channels are present.
    """
    if not channel_labels:
        return False

    non_aux = [
        label
        for label in channel_labels
        if _categorize_channel(label) in {"Other", "EEG"}
    ]
    if not non_aux:
        return False

    numeric_count = sum(1 for label in non_aux if _is_numeric_style_channel_label(label))
    if numeric_count == 0:
        return False

    # If we already have explicit non-numeric non-EEG labels (e.g. VST 01,
    # matte 01), this is often not the numeric-ID failure mode. However some
    # recordings have a single named auxiliary/marker channel plus a large block
    # of numeric EEG contacts; in that case recovery is still appropriate.
    non_numeric_other = [
        label
        for label in non_aux
        if _categorize_channel(label) == "Other" and not _is_numeric_style_channel_label(label)
    ]
    if non_numeric_other and (numeric_count / max(1, len(non_aux))) < 0.75:
        return False

    # Primary target: all labels are numeric-style.
    if numeric_count == len(non_aux):
        return True

    # Also recover partial cases where labels are predominantly numeric and only
    # a subset already resolved as EEG.
    threshold = max(2, (len(non_aux) + 1) // 2)
    return numeric_count >= threshold


def _recover_channel_labels_from_montage(header, channel_labels: list[str]) -> list[str]:
    """Recover numeric placeholder labels using available montage metadata."""
    if not channel_labels:
        return channel_labels
    if not _should_attempt_numeric_montage_recovery(channel_labels):
        return channel_labels

    sse_fixed_map_recovery = _recover_sse_64_numeric_id_labels(header, channel_labels)
    if sse_fixed_map_recovery is not None:
        return sse_fixed_map_recovery

    montage_entries = getattr(header, "MontageInfo", None) or []
    if not montage_entries:
        return channel_labels

    recovery_plan = build_montage_recovery_plan(montage_entries, channel_labels)
    primary_entries = recovery_plan.primary_entries
    aux_entries = recovery_plan.auxiliary_entries
    has_unknown_catalog = recovery_plan.has_unknown_catalog

    signal_to_electrode: dict[str, str] = {}
    conflicts: set[str] = set()

    def _collect_mappings(
        entries: list[dict[str, object]],
        mapping: dict[str, str],
        mapping_conflicts: set[str],
    ) -> None:
        for entry in entries:
            signal_name = _clean_channel_label(entry.get("signalName1", ""))
            signal_name_2 = _clean_channel_label(entry.get("signalName2", ""))
            derivation_name = entry.get("derivationName", "")
            source = str(entry.get("source", ""))
            if not signal_name or not derivation_name:
                continue
            if source in DIRECT_ID_NAME_SOURCES:
                fixed_name = _clean_channel_label(derivation_name)
                fixed_upper = fixed_name.upper()
                # Only treat these sources as direct ID->name tables when the
                # row label is a simple contact/electrode name. Some fixed
                # DERIVATION tables contain derivation labels (e.g. "Fp2-av")
                # that must be parsed as endpoints instead.
                direct_name_usable = True
                if source == "derivation_fixed_table" and any(
                    sep in fixed_name for sep in ("-", "/", "\\", "+", ":")
                ):
                    direct_name_usable = False
                # Hidden catalogs can contain placeholder rows (AV/REF) mixed
                # with useful labels. Do not use those as final channel names.
                if source == "unknown_montage_catalog" and (
                    fixed_upper in {"AV", "REF", "REFERENCE"}
                    or _NUMERIC_REF_LABEL_RE.fullmatch(fixed_upper)
                ):
                    direct_name_usable = False
                if source == "unknown_montage_catalog" and any(
                    sep in fixed_name for sep in ("-", "/", "\\", "+", ":")
                ):
                    # Hidden catalogs can include derived rows such as "Fp2-av".
                    # Keep those out of direct ID->contact mapping.
                    direct_name_usable = False
                if direct_name_usable:
                    # Fixed tables may legitimately contain numeric contact labels
                    # (e.g. depth contacts "1..32"), so we allow both numeric and
                    # non-numeric names here and let identity mappings be harmless.
                    _add_mapping_with_conflict_tracking(
                        mapping, mapping_conflicts, signal_name, fixed_name or None
                    )
                    continue
                if source == "unknown_montage_catalog" and any(
                    sep in fixed_name for sep in ("-", "/", "\\", "+", ":")
                ):
                    # For hidden catalogs, derived labels are noisy and can map
                    # placeholders to the wrong endpoint if we parse them as
                    # pair derivations. Skip these rows entirely.
                    continue
            left, right = _extract_eeg_endpoints_from_derivation(derivation_name)
            if left is None:
                left = _extract_non_eeg_single_derivation_label(derivation_name)
            if (
                left is None
                and source == "derivation_main_table"
                and not signal_name_2
            ):
                # Main DERIVATION tables in some recordings contain direct
                # signal-ID -> channel-name rows (e.g. VTP1, HST3, F3, CZ)
                # rather than pair derivations. Use the row name directly.
                direct_name = _clean_channel_label(derivation_name)
                if should_skip_main_direct_numeric_ref_fallback(
                    source, direct_name, has_unknown_catalog=has_unknown_catalog
                ):
                    direct_name = ""
                left = direct_name or None
            _add_mapping_with_conflict_tracking(mapping, mapping_conflicts, signal_name, left)
            # Some numeric channels only appear as signalName2 (right endpoint).
            if signal_name_2 and signal_name_2.isdigit():
                _add_mapping_with_conflict_tracking(
                    mapping, mapping_conflicts, signal_name_2, right
                )

    # Primary montage rows define the authoritative mapping.
    _collect_mappings(primary_entries, signal_to_electrode, conflicts)
    for key in conflicts:
        signal_to_electrode.pop(key, None)

    # Auxiliary rows (e.g. hidden input-montage tables) only fill gaps and
    # never override primary mappings.
    if aux_entries:
        def _aux_group_key(entry: dict[str, object]) -> tuple[str, str]:
            source = str(entry.get("source", ""))
            if source == "unknown_montage_catalog":
                return source, str(entry.get("montageName", "") or "")
            return source, ""

        grouped_aux_entries: dict[tuple[str, str], list[dict[str, object]]] = {}
        for entry in aux_entries:
            key = _aux_group_key(entry)
            if key not in grouped_aux_entries:
                grouped_aux_entries[key] = []
            grouped_aux_entries[key].append(entry)

        for group_entries in grouped_aux_entries.values():
            aux_mapping: dict[str, str] = {}
            aux_conflicts: set[str] = set()
            _collect_mappings(group_entries, aux_mapping, aux_conflicts)
            for key in aux_conflicts:
                aux_mapping.pop(key, None)
            for key, value in aux_mapping.items():
                if key not in signal_to_electrode:
                    signal_to_electrode[key] = value

    if not signal_to_electrode:
        return channel_labels

    recovered_labels: list[str] = []
    replaced = 0
    for label in channel_labels:
        cleaned = _clean_channel_label(label)
        normalized = _normalize_eeg_position(cleaned)
        if normalized:
            recovered_labels.append(normalized)
            continue
        mapped = signal_to_electrode.get(_channel_match_key(cleaned))
        is_placeholder = bool(cleaned.isdigit() or _NUMERIC_REF_LABEL_RE.fullmatch(cleaned.upper()))
        if mapped and is_placeholder:
            recovered_labels.append(mapped)
            replaced += 1
        else:
            recovered_labels.append(cleaned)

    eeg_before = sum(1 for label in channel_labels if _categorize_channel(label) == "EEG")
    eeg_after = sum(1 for label in recovered_labels if _categorize_channel(label) == "EEG")
    if eeg_after < eeg_before:
        return channel_labels
    if replaced > 0:
        logger.debug(
            "Montage recovery source counts=%s source_scores=%s catalog_scores=%s",
            recovery_plan.source_counts,
            recovery_plan.source_scores,
            recovery_plan.catalog_scores,
        )
        logger.info(
            "Recovered %d/%d channel labels from montage derivations",
            replaced,
            len(channel_labels),
        )
    return recovered_labels


def _channel_labels(header, channels: Iterable[int]) -> list[str]:
    channel_list = list(channels)
    labels = header.Segments[0].chName if header.Segments else []
    resolved: list[str] = []
    for channel in channel_list:
        zero_based = channel - 1
        try:
            label = labels[zero_based]
        except (IndexError, TypeError):
            label = f"Ch{channel}"
        cleaned = _clean_channel_label(label)
        resolved.append(cleaned or f"Ch{channel}")
    return _disambiguate_duplicate_channel_labels(header, channel_list, resolved)


def _channel_references(header, channels: list[int]) -> list[str]:
    segment_refs = header.Segments[0].refName if header.Segments else []
    ts_entries = header.TSInfo or []
    references: list[str] = []
    for channel in channels:
        zero_based = channel - 1
        ref_label = ""
        try:
            ref_label = segment_refs[zero_based]
        except (IndexError, TypeError):
            ref_label = ""
        cleaned = _clean_channel_label(ref_label)
        if not cleaned and 0 <= zero_based < len(ts_entries):
            cleaned = _clean_channel_label(getattr(ts_entries[zero_based], "refSensor", ""))
        references.append(cleaned)
    return references


def _ensure_unique_labels(channel_labels: list[str]) -> list[str]:
    unique_labels: list[str] = []
    used: set[str] = set()
    next_suffix: dict[str, int] = {}

    for label in channel_labels:
        base = _clean_channel_label(label) or "Ch"
        candidate = base
        suffix = next_suffix.get(base, 2)
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        next_suffix[base] = suffix
        used.add(candidate)
        unique_labels.append(candidate)
    return unique_labels


def _disambiguate_duplicate_channel_labels(
    header,
    channels: list[int],
    channel_labels: list[str],
) -> list[str]:
    if not channel_labels:
        return channel_labels

    label_positions: dict[str, list[int]] = {}
    for idx, label in enumerate(channel_labels):
        label_positions.setdefault(label, []).append(idx)

    duplicates = {label: idxs for label, idxs in label_positions.items() if len(idxs) > 1}
    if not duplicates:
        return channel_labels

    references = _channel_references(header, channels)
    resolved = list(channel_labels)

    for label, idxs in duplicates.items():
        refs_for_label = {_clean_channel_label(references[idx]).upper() for idx in idxs if references[idx]}
        if len(refs_for_label) < 2:
            continue
        for idx in idxs:
            ref = _clean_channel_label(references[idx])
            if ref:
                resolved[idx] = f"{label}-{ref}"

    return _ensure_unique_labels(resolved)


def _select_channels(header, *, include_all: bool = False) -> list[int]:
    if include_all:
        if header.ChannelInfo and header.Segments:
            label_count = len(header.Segments[0].chName) if header.Segments else 0
            if label_count and len(header.ChannelInfo) == label_count:
                channels = [
                    idx + 1 for idx, info in enumerate(header.ChannelInfo) if info.get("bOn", True)
                ]
                if channels:
                    return channels
        if header.TSInfo:
            return list(range(1, len(header.TSInfo) + 1))
        if header.Segments and header.Segments[0].chName:
            return list(range(1, len(header.Segments[0].chName) + 1))
    if header.matchingChannels:
        return list(header.matchingChannels)
    if header.TSInfo:
        return list(range(1, len(header.TSInfo) + 1))
    if header.Segments and header.Segments[0].chName:
        return list(range(1, len(header.Segments[0].chName) + 1))
    return []


def _channel_rate_and_variation(header, channel_zero_based: int) -> tuple[float | None, bool]:
    if not header.Segments:
        return None, False
    rates: list[float] = []
    for segment in header.Segments:
        if segment.samplingRate is None:
            continue
        if channel_zero_based < len(segment.samplingRate):
            rates.append(float(segment.samplingRate[channel_zero_based]))
    if not rates:
        return None, False
    base = rates[0]
    varied = any(not np.isclose(rate, base, rtol=1e-6, atol=1e-6) for rate in rates[1:])
    return base, varied


def _sampling_rate_variation(header, channels: Iterable[int]) -> tuple[bool, bool]:
    rates: list[float] = []
    mixed_over_time = False
    for ch in channels:
        rate, varied = _channel_rate_and_variation(header, ch - 1)
        if rate is not None:
            rates.append(rate)
        if varied:
            mixed_over_time = True
    mixed_across_channels = False
    if rates:
        rounded = np.round(np.array(rates, dtype=float), decimals=6)
        mixed_across_channels = len(np.unique(rounded)) > 1
    return mixed_across_channels, mixed_over_time


def _require_scipy_resample():
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Resampling requires scipy. Install it with: uv pip install -p .venv scipy"
        ) from exc
    return resample_poly


def _resample_ratio(source_len: int, target_len: int, max_den: int = 1000) -> tuple[int, int]:
    if source_len <= 0 or target_len <= 0:
        raise ValueError("Resample lengths must be positive")
    ratio = target_len / source_len
    fraction = Fraction(ratio).limit_denominator(max_den)
    return fraction.numerator, fraction.denominator


def _trim_or_pad(data: np.ndarray, target_len: int, *, axis: int = 0) -> np.ndarray:
    current = data.shape[axis]
    if current == target_len:
        return data
    if current > target_len:
        slicer = [slice(None)] * data.ndim
        slicer[axis] = slice(0, target_len)
        return data[tuple(slicer)]
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, target_len - current)
    return np.pad(data, pad_width, mode="constant")


def _resample_waveform(waveform: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if target_fs <= 0:
        raise ValueError("Resample rate must be positive")
    if np.isclose(original_fs, target_fs):
        return waveform

    n_samples, n_channels = waveform.shape
    duration_seconds = n_samples / float(original_fs)
    target_samples = max(int(round(duration_seconds * target_fs)), 1)
    if waveform.size == 0:
        return np.zeros((target_samples, n_channels), dtype=waveform.dtype)
    if n_samples == 1:
        return np.repeat(waveform, target_samples, axis=0).astype(waveform.dtype, copy=False)

    resample_poly = _require_scipy_resample()
    up, down = _resample_ratio(n_samples, target_samples)
    resampled = resample_poly(waveform, up, down, axis=0, padtype="reflect")
    resampled = _trim_or_pad(resampled, target_samples, axis=0)
    return resampled.astype(waveform.dtype, copy=False)


def _resample_vector(
    samples: np.ndarray,
    original_fs: float,
    target_fs: float,
    target_samples: int | None = None,
) -> np.ndarray:
    if target_fs <= 0:
        raise ValueError("Resample rate must be positive")
    if original_fs <= 0:
        raise ValueError("Original sampling rate must be positive")
    if target_samples is None:
        duration_seconds = len(samples) / float(original_fs) if original_fs else 0.0
        target_samples = max(int(round(duration_seconds * target_fs)), 1)
    if target_samples <= 0:
        return np.empty(0, dtype=np.float32)
    if np.isclose(original_fs, target_fs) and target_samples == len(samples):
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if len(samples) == 1:
        return np.full(target_samples, float(samples[0]), dtype=np.float32)
    resample_poly = _require_scipy_resample()
    up, down = _resample_ratio(len(samples), target_samples)
    resampled = resample_poly(samples, up, down, axis=0, padtype="reflect")
    resampled = _trim_or_pad(resampled, target_samples, axis=0)
    return resampled.astype(np.float32)


def _segment_target_samples(header, target_fs: float) -> list[int]:
    durations = [float(seg.duration) for seg in header.Segments] if header.Segments else []
    if not durations:
        return []
    total_duration = sum(durations)
    total_target = max(int(round(total_duration * target_fs)), 1)
    per_segment = [max(int(round(duration * target_fs)), 0) for duration in durations]
    if per_segment:
        per_segment[-1] = max(total_target - sum(per_segment[:-1]), 0)
    return per_segment


def _segment_sample_window(header, channel_zero_based: int, seg_idx: int) -> tuple[int, int] | None:
    if not header.Segments:
        return None
    if seg_idx < 0 or seg_idx >= len(header.Segments):
        return None
    offset = 0
    for seg in header.Segments[:seg_idx]:
        if seg.sampleCount is None or channel_zero_based >= len(seg.sampleCount):
            return None
        offset += int(seg.sampleCount[channel_zero_based])
    target_seg = header.Segments[seg_idx]
    if target_seg.sampleCount is None or channel_zero_based >= len(target_seg.sampleCount):
        return None
    count = int(target_seg.sampleCount[channel_zero_based])
    if count <= 0:
        return None
    return offset + 1, offset + count


def _read_segment_waveform(
    input_path: Path,
    header,
    channels: list[int],
    seg_idx: int,
) -> np.ndarray:
    if not channels:
        return np.zeros((0, 0), dtype=np.float32)
    base_channel = channels[0] - 1
    window = _segment_sample_window(header, base_channel, seg_idx)
    if window:
        beg, end = window
        try:
            data = read_nervus_data(input_path, header, channels=channels, begsample=beg, endsample=end)
            return data.T
        except NotImplementedError:
            pass
    # Fallback: read per channel and pad to max length.
    samples: list[np.ndarray] = []
    max_len = 0
    for ch in channels:
        window = _segment_sample_window(header, ch - 1, seg_idx)
        if not window:
            vec = np.zeros(0, dtype=np.float32)
        else:
            beg, end = window
            data = read_nervus_data(input_path, header, channels=[ch], begsample=beg, endsample=end)
            vec = data[0] if data.size else np.zeros(0, dtype=np.float32)
        samples.append(vec)
        max_len = max(max_len, vec.size)
    waveform = np.zeros((max_len, len(channels)), dtype=np.float32)
    for idx, vec in enumerate(samples):
        if vec.size:
            waveform[: vec.size, idx] = vec
    return waveform


def _segment_events(
    events: Sequence[EventItem] | None,
    segment: SegmentInfo,
    seg_idx: int,
) -> list[EventItem]:
    if not events or not segment.date or not segment.duration:
        return []
    seg_start = segment.date
    seg_end = seg_start + timedelta(seconds=float(segment.duration))
    selected: list[EventItem] = []
    for event in events:
        if event.segmentIndex == seg_idx:
            selected.append(event)
            continue
        if event.date and seg_start <= event.date < seg_end:
            selected.append(event)
    return selected


_SLEEP_STAGE_NAMES = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage N3",  # R&K stage 4 → AASM N3
    5: "Sleep stage R",
}


def _sleep_stage_events(
    sleep_scores: list[SleepScoreInfo],
    recording_start: datetime,
) -> list[EventItem]:
    """Convert the primary hypnogram to a list of EventItems for EDF+ export."""
    if not sleep_scores or recording_start is None:
        return []
    primary = sleep_scores[0]
    events: list[EventItem] = []
    for epoch in primary.epochs:
        onset_s = (epoch.index - 1) * epoch.duration
        try:
            epoch_dt = recording_start + timedelta(seconds=onset_s)
        except (OverflowError, ValueError):
            logger.warning("Skipping sleep epoch %d: onset %ds out of range", epoch.index, onset_s)
            continue
        label = _SLEEP_STAGE_NAMES.get(epoch.stage, f"Sleep stage {epoch.stage}")
        events.append(
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=epoch_dt,
                duration=float(epoch.duration),
                user="",
                GUID="",
                label=label,
                IDStr="",
            )
        )
    return events


def _segment_output_path(base_output: Path, seg_idx: int, seg_count: int) -> Path:
    if seg_count <= 1:
        return base_output
    return base_output.with_name(f"{base_output.stem}_seg{seg_idx + 1}{base_output.suffix}")


def _filter_vendor_events(events: Sequence[EventItem] | None) -> list[EventItem]:
    if not events:
        return []
    skip_guids = {
        "{93A2CB2C-F420-4672-AA62-18989F768519}",  # Detections Inactive
        "{98FB933E-5183-4E4D-99AF-88AA29B22D05}",  # Detections Active
        "{96315D79-5C24-4A65-B334-E31A95088D55}",  # Us. start
        "{725798BF-CD1C-4909-B793-6C7864C27AB7}",  # Review progress
    }
    skip_ids = {
        "Video Review Progress",
        "Exam start",
        "Video Start",
        "Video Stop",
    }
    return [
        event
        for event in events
        if event.GUID not in skip_guids and event.IDStr not in skip_ids
    ]


def _read_channels_native_multirate(
    input_path: Path,
    header,
    channels: list[int],
) -> tuple[list[np.ndarray], list[float]]:
    """Read each channel at its native sampling rate without resampling.

    Returns a tuple of (channel_data_list, channel_rates_list) where each
    element corresponds to one channel in *channels*.  Channels with
    variable rates across segments are read at the first-segment rate.
    """
    channel_data: list[np.ndarray] = []
    channel_rates: list[float] = []
    for ch in channels:
        rate, _ = _channel_rate_and_variation(header, ch - 1)
        if rate is None or rate <= 0:
            rate = float(header.targetSamplingRate or 0.0)
        data = read_nervus_data(input_path, header, channels=[ch])
        vec: np.ndarray = data[0] if data.size else np.zeros(0, dtype=np.float32)
        channel_data.append(vec)
        channel_rates.append(float(rate))
    return channel_data, channel_rates


def _resample_channel_segments(
    input_path: Path,
    header,
    channel: int,
    target_fs: float,
    target_segments: list[int],
) -> np.ndarray:
    channel_zb = channel - 1
    samples_out: list[np.ndarray] = []
    offset = 0
    for seg_idx, segment in enumerate(header.Segments):
        seg_samples = 0
        if segment.sampleCount is not None and channel_zb < len(segment.sampleCount):
            seg_samples = int(segment.sampleCount[channel_zb])
        target_len = target_segments[seg_idx] if seg_idx < len(target_segments) else 0
        if seg_samples <= 0 or target_len <= 0:
            samples_out.append(np.zeros(target_len, dtype=np.float32))
            offset += max(seg_samples, 0)
            continue
        beg = offset + 1
        end = offset + seg_samples
        data = read_nervus_data(input_path, header, channels=[channel], begsample=beg, endsample=end)
        samples = data[0] if data.size else np.zeros(0, dtype=np.float32)
        rate = None
        if segment.samplingRate is not None and channel_zb < len(segment.samplingRate):
            rate = float(segment.samplingRate[channel_zb])
        if rate is None or rate <= 0:
            rate = float(header.targetSamplingRate or 0.0)
        if rate <= 0:
            samples_out.append(np.zeros(target_len, dtype=np.float32))
        else:
            samples_out.append(_resample_vector(samples, rate, target_fs, target_len))
        offset += seg_samples
    if not samples_out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(samples_out)


def _read_and_resample_mixed(
    input_path: Path,
    header,
    channels: list[int],
    target_fs: float,
    segment_aware: bool,
) -> np.ndarray:
    waveform = None
    target_segments = _segment_target_samples(header, target_fs) if segment_aware else []
    total_target = sum(target_segments) if target_segments else None
    for idx, ch in enumerate(channels):
        rate, varied = _channel_rate_and_variation(header, ch - 1)
        if rate is None or rate <= 0:
            raise RuntimeError(f"Unable to determine sampling rate for channel {ch}")
        if varied and not segment_aware:
            logger.warning(
                "Sampling rate changes across segments for channel %d; using first-segment rate %.4f Hz",
                ch,
                rate,
            )
        if segment_aware and target_segments:
            resampled = _resample_channel_segments(
                input_path, header, ch, target_fs, target_segments
            )
            target_samples = resampled.size
        else:
            data = read_nervus_data(input_path, header, channels=[ch])
            samples = data[0] if data.size else np.zeros(0, dtype=np.float32)
            if total_target is None:
                duration = len(samples) / float(rate) if rate else 0.0
                total_target = max(int(round(duration * target_fs)), 1)
            target_samples = total_target
            resampled = _resample_vector(samples, rate, target_fs, target_samples)
        if waveform is None:
            waveform = np.zeros((target_samples, len(channels)), dtype=np.float32)
        waveform[: resampled.size, idx] = resampled
    if waveform is None:
        return np.zeros((0, 0), dtype=np.float32)
    return waveform


def _bandpass_filter(
    waveform: np.ndarray,
    fs: float,
    lowcut: float | None,
    highcut: float | None,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass/highpass/lowpass filter.
    
    Args:
        waveform: Shape (samples, channels) array of signal data.
        fs: Sampling frequency in Hz.
        lowcut: High-pass cutoff frequency in Hz (or None to skip).
        highcut: Low-pass cutoff frequency in Hz (or None to skip).
    
    Returns:
        Filtered waveform with same shape as input.
    """
    # If no filtering requested, return as-is
    if lowcut is None and highcut is None:
        return waveform
    
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:
        raise RuntimeError(
            "Filtering requires scipy. Install it with: uv pip install -p .venv scipy"
        ) from exc
    
    nyquist = fs / 2.0
    order = 4  # Standard order for EEG filtering
    
    # Determine filter type and normalized frequencies
    if lowcut is not None and highcut is not None:
        # Bandpass filter
        low = lowcut / nyquist
        high = highcut / nyquist
        # Clamp to valid range (0, 1)
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        if low >= high:
            raise ValueError(f"lowcut ({lowcut} Hz) must be less than highcut ({highcut} Hz)")
        b, a = butter(order, [low, high], btype="band")
    elif lowcut is not None:
        # High-pass filter only
        low = lowcut / nyquist
        low = max(0.001, min(low, 0.999))
        b, a = butter(order, low, btype="high")
    else:
        # Low-pass filter only (highcut is not None)
        high = highcut / nyquist
        high = max(0.001, min(high, 0.999))
        b, a = butter(order, high, btype="low")
    
    # Apply zero-phase filtering to each channel
    _, n_channels = waveform.shape
    filtered = np.zeros_like(waveform)
    for idx in range(n_channels):
        filtered[:, idx] = filtfilt(b, a, waveform[:, idx])
    
    return filtered


def _notch_filter(
    waveform: np.ndarray,
    fs: float,
    notch_freq: float | None,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply a zero-phase notch filter to remove powerline interference.
    
    Args:
        waveform: Shape (samples, channels) array of signal data.
        fs: Sampling frequency in Hz.
        notch_freq: Center frequency to notch out (e.g. 50 or 60 Hz), or None to skip.
        quality_factor: Quality factor Q (higher = narrower notch). Default 30.
    
    Returns:
        Filtered waveform with same shape as input.
    """
    if notch_freq is None or notch_freq <= 0:
        return waveform
    
    try:
        from scipy.signal import iirnotch, filtfilt
    except ImportError as exc:
        raise RuntimeError(
            "Filtering requires scipy. Install it with: uv pip install -p .venv scipy"
        ) from exc
    
    # Design notch filter
    w0 = notch_freq / (fs / 2.0)  # Normalized frequency
    if w0 >= 1.0:
        # Notch frequency is at or above Nyquist - skip
        return waveform
    
    b, a = iirnotch(w0, quality_factor)
    
    # Apply zero-phase filtering to each channel
    _, n_channels = waveform.shape
    filtered = np.zeros_like(waveform)
    for idx in range(n_channels):
        filtered[:, idx] = filtfilt(b, a, waveform[:, idx])
    
    return filtered


def _ecg_channel_indices(channel_labels: list[str]) -> list[int]:
    """Return indices of channels identified as ECG/EKG."""
    return [
        i for i, label in enumerate(channel_labels)
        if label.upper().strip() in ("EKG", "ECG")
    ]


def _apply_ecg_polarity_correction(
    data: np.ndarray | list[np.ndarray],
    ecg_indices: list[int],
) -> np.ndarray | list[np.ndarray]:
    """Negate ECG channels to correct for Nicolet amplifier polarity convention.

    Nicolet EEG amplifiers record ECG with reversed input polarity compared to
    the standard Lead II convention (V_RA−V_LL instead of V_LL−V_RA).  The
    Nicolet viewer compensates during display; EDF writers must negate these
    channels so that downstream EDF tools see the correct polarity.
    """
    if not ecg_indices:
        return data
    if isinstance(data, list):
        result = list(data)
        for i in ecg_indices:
            result[i] = -result[i]
        return result
    # 2-D array: shape (samples, channels)
    corrected = data.copy()
    for i in ecg_indices:
        corrected[:, i] = -corrected[:, i]
    return corrected


def _categorize_channel(label: str) -> str:
    """Categorize a channel label into a type based on naming conventions."""
    label_upper = _clean_channel_label(label).upper()
    base_label = re.sub(r"_\d+$", "", label_upper)
    
    # EOG (electrooculogram) channels
    if base_label in ("ROC", "LOC", "EOG", "HEOG", "VEOG", "LEOG", "REOG"):
        return "EOG"
    
    # EKG/ECG (electrocardiogram) channels
    if base_label in ("EKG", "ECG"):
        return "EKG"
    
    # Stimulus/trigger channels
    if base_label in ("PHOTIC", "STIM", "TRIGGER", "TRIG"):
        return "Stimulus"
    
    # Reference channels (ear references, not midline EEG)
    if base_label in ("A1", "A2", "REF", "REFERENCE", "GROUND", "GND"):
        return "Reference"

    # Common EEG variants not covered by the standard position normalizer.
    # Pg1/Pg2 are posterior electrodes that appear in real KNF files.
    if base_label in ("PG1", "PG2"):
        return "EEG"
    
    if _normalize_eeg_position(base_label):
        return "EEG"
    if _extract_eeg_label_from_derivation(base_label):
        return "EEG"
    
    # If we can't categorize it, it's "Other"
    return "Other"


def _build_channels_list(
    channel_labels: list[str],
) -> list[dict[str, object]]:
    """Build a flat list of channels with type info for easy DataFrame conversion."""
    channels = []
    for idx, label in enumerate(channel_labels):
        item = {
            "index": idx,
            "name": label,
            "type": _categorize_channel(label),
        }
        channels.append(item)
    return channels


def _count_channels_by_type(channel_labels: list[str]) -> dict[str, int]:
    """Count channels by type for flat access."""
    counts: dict[str, int] = {}
    for label in channel_labels:
        ch_type = _categorize_channel(label)
        counts[ch_type] = counts.get(ch_type, 0) + 1
    return counts


def _write_json_sidecar(
    output_path: Path,
    *,
    source_path: Path,
    sampling_rate: float,
    channel_labels: list[str],
    sample_count: int,
    start_time: datetime | None,
    events: Sequence[EventItem] | None,
    nrv_header: NervusHeader,
) -> None:
    """Write ML-friendly JSON sidecar with recording metadata.
    
    Structure is optimized for machine learning pipelines:
    - Flat top-level fields for easy filtering/indexing
    - Channels as list of {index, name, type} for DataFrame conversion
    - Events separated into annotations vs system events
    - All times in seconds relative to recording start
    """
    start_iso = start_time.isoformat() if start_time else None
    
    # Calculate duration
    duration_seconds = sample_count / sampling_rate if sampling_rate > 0 else None
    
    # Build channel type counts (flat, for easy filtering)
    type_counts = _count_channels_by_type(channel_labels)
    
    # Process events - separate annotations from system events
    annotations = []
    system_events = []
    
    if events:
        for event in events:
            onset = None
            if start_time and event.date:
                onset = (event.date - start_time).total_seconds()
            
            event_id, label, annotation = canonical_event_text(event)
            
            # Separate annotations (with text) from system events
            if annotation:
                clean_text = annotation.strip()
                if clean_text:
                    annotations.append({
                        "onset_seconds": onset,
                        "duration_seconds": event.duration if event.duration else None,
                        "text": clean_text,
                    })
            else:
                # Skip noisy events: UNKNOWN type with placeholder label "-" or no label
                # These are typically review/selection markers without clinical value
                if event_id == "UNKNOWN" and (label == "-" or label is None):
                    continue
                
                # For meaningful system events, don't include user (reduces noise)
                system_events.append({
                    "onset_seconds": onset,
                    "duration_seconds": event.duration if event.duration else None,
                    "type": event_id,
                    "label": label if label != "-" else None,
                })
    
    # Build the payload with ML-friendly structure
    payload = {
        # ===== Recording identification =====
        "source_file": str(source_path.resolve()),  # Full absolute path to original Nicolet file
        "edf_file": str(output_path.resolve()),  # Full absolute path to generated EDF file
        
        # ===== Signal properties (flat, for filtering) =====
        "sampling_rate_hz": sampling_rate,
        "sample_count": sample_count,
        "duration_seconds": duration_seconds,
        "start_time": start_iso,
        
        # ===== Channel summary (flat counts for easy filtering) =====
        "channel_count": len(channel_labels),
        "eeg_channel_count": type_counts.get("EEG", 0),
        "eog_channel_count": type_counts.get("EOG", 0),
        "ekg_channel_count": type_counts.get("EKG", 0),
        "other_channel_count": type_counts.get("Other", 0) + type_counts.get("Reference", 0) + type_counts.get("Stimulus", 0),
        
        # ===== Channels list (for DataFrame: pd.DataFrame(data["channels"])) =====
        "channels": _build_channels_list(channel_labels),
        
        # ===== Recording metadata =====
        "reference": nrv_header.reference,
        "n_segments": len(nrv_header.Segments) if nrv_header.Segments else 0,
        "excluded_channels": nrv_header.excludedChannels if nrv_header.excludedChannels else [],
        # ===== Clinical annotations (for NLP/labeling) =====
        "annotations": annotations,
        "annotation_count": len(annotations),
        
        # ===== System events (for QC/preprocessing) =====
        "events": system_events,
        "event_count": len(system_events),
    }
    
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(payload, indent=2))


def _adjust_events_for_gaps(
    events: Sequence[EventItem] | None,
    segments: Sequence[SegmentInfo] | None,
    start_time: datetime | None,
) -> list[EventItem]:
    if not events or not segments or not start_time:
        return list(events) if events else []
    skip_guids = {
        "{93A2CB2C-F420-4672-AA62-18989F768519}",  # Detections Inactive
    }
    segment_starts = []
    cumulative = 0.0
    for seg in segments:
        segment_starts.append(cumulative)
        cumulative += float(seg.duration)

    video_guids = {
        "{32F2469E-6792-4CAD-8E11-B7747688BB8B}",
        "{056F522F-DDA5-48B9-82E1-1A75C35CBC30}",
    }
    video_count = sum(1 for ev in events if ev.GUID in video_guids)

    adjusted = []
    seen = set()
    for event in events:
        if event.GUID in skip_guids:
            continue
        if event.IDStr == "Recording Paused":
            continue
        if event.IDStr in {"Video Review Progress", "Exam start"}:
            continue
        if event.GUID in video_guids and video_count < 10:
            continue
        if event.IDStr == "Us. start" and video_count < 10:
            continue
        if event.IDStr == "UNKNOWN":
            cleaned = event.label.strip() if event.label else ""
            if not cleaned or cleaned == "-":
                continue
        if event.segmentIndex is None or event.segmentIndex >= len(segments):
            adjusted.append(event)
            continue
        seg = segments[event.segmentIndex]
        if not event.date or not seg.date:
            adjusted.append(event)
            continue
        offset = (event.date - seg.date).total_seconds()
        if offset < 0:
            offset = 0.0
        onset = segment_starts[event.segmentIndex] + offset
        if event.IDStr == "Review progress" and onset > 1.0:
            continue
        new_date = start_time + timedelta(seconds=onset)
        # Some legacy files label the initial review marker as "Review progress".
        if onset <= 1.0 and event.IDStr.lower() == "review progress":
            event_label = None
            event_id = "Us. start"
            event_duration = 0.0
            if video_count < 10:
                continue
        else:
            event_label = event.label
            event_id = event.IDStr
            event_duration = event.duration
        adjusted.append(
            EventItem(
                dateOLE=event.dateOLE,
                dateFraction=event.dateFraction,
                date=new_date,
                duration=event_duration,
                user=event.user,
                GUID=event.GUID,
                label=event_label,
                IDStr=event_id,
                annotation=event.annotation,
                segmentIndex=event.segmentIndex,
                isEpoch=event.isEpoch,
                rawLabel=event.rawLabel,
            )
        )
        key = (event_id, round(onset, 3), event_label or "")
        if key in seen:
            adjusted.pop()
            continue
        seen.add(key)
    return adjusted


def convert_file(
    input_path: Path,
    output_dir: Path,
    patient_rules: list[dict[str, str]],
    *,
    output_path: Path | None = None,
    input_root: Path | None = None,
    json_sidecar: bool = False,
    resample_to: float | None = None,
    lowcut: float | None = None,
    highcut: float | None = None,
    notch: float | None = None,
    split_by_segment: bool = False,
    vendor_style: bool = False,
    status_cb=None,
) -> Path:
    if status_cb:
        status_cb("read header")
    public_header, nrv_header = read_nervus_header(input_path)
    fs = public_header.get("Fs") or nrv_header.targetSamplingRate
    if not fs:
        raise RuntimeError("Unable to determine sampling frequency from header")
    fs = float(fs)
    if resample_to is None:
        _require_integer_output_rate(fs, context="Input")

    channels = _select_channels(nrv_header, include_all=True)
    if not channels:
        raise RuntimeError("No channels available for conversion")

    if status_cb:
        status_cb("read data")
    channel_labels = _channel_labels(nrv_header, channels)
    ecg_indices = _ecg_channel_indices(channel_labels)
    recovered_channel_labels = _recover_channel_labels_from_montage(nrv_header, channel_labels)
    mixed_across_channels, mixed_over_time = _sampling_rate_variation(nrv_header, channels)
    if resample_to is not None and mixed_over_time:
        logger.warning(
            "Sampling rates vary across segments; resampling is applied per segment."
        )

    patient_meta = _select_patient_metadata(input_path, patient_rules, nrv_header)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = Path(output_path) if output_path else _candidate_output_path(input_path, output_dir, input_root)
    segments = nrv_header.Segments or []
    if split_by_segment and len(segments) > 1:
        for seg_idx, segment in enumerate(segments):
            if status_cb:
                status_cb(f"read data (segment {seg_idx + 1}/{len(segments)})")
            seg_waveform = _read_segment_waveform(input_path, nrv_header, channels, seg_idx)
            seg_fs = fs
            if segment.samplingRate is not None and segment.samplingRate.size:
                ch0 = channels[0] - 1
                if ch0 < len(segment.samplingRate):
                    seg_fs = float(segment.samplingRate[ch0])
            if resample_to is None:
                _require_integer_output_rate(seg_fs, context=f"Segment {seg_idx + 1}")
            if resample_to is not None:
                if status_cb:
                    status_cb(f"resample (segment {seg_idx + 1}/{len(segments)})")
                seg_waveform = _resample_waveform(seg_waveform, float(seg_fs), float(resample_to))
                seg_fs = float(resample_to)
            if lowcut is not None or highcut is not None:
                if status_cb:
                    status_cb(f"bandpass filter (segment {seg_idx + 1}/{len(segments)})")
                seg_waveform = _bandpass_filter(seg_waveform, float(seg_fs), lowcut, highcut)
            if notch is not None and notch > 0:
                if status_cb:
                    status_cb(f"notch filter (segment {seg_idx + 1}/{len(segments)})")
                seg_waveform = _notch_filter(seg_waveform, float(seg_fs), notch)
            seg_waveform = _apply_ecg_polarity_correction(seg_waveform, ecg_indices)

            segment_events = _segment_events(nrv_header.Events, segment, seg_idx)
            if vendor_style:
                segment_events = _filter_vendor_events(segment_events)
            segment_events = normalize_events(segment_events)
            segment_output = _segment_output_path(resolved_output, seg_idx, len(segments))
            segment_start = segment.date or nrv_header.startDateTime
            sleep_events = _sleep_stage_events(nrv_header.SleepScores, nrv_header.startDateTime)
            if sleep_events and segment_start and segment.duration:
                seg_end = segment_start + timedelta(seconds=float(segment.duration))
                sleep_events = [e for e in sleep_events if segment_start <= e.date < seg_end]
            segment_events = list(segment_events) + sleep_events
            if status_cb:
                status_cb(f"write edf (segment {seg_idx + 1}/{len(segments)})")
            write_edf(
                segment_output,
                seg_waveform,
                seg_fs,
                recovered_channel_labels,
                patient_meta,
                recording_start=segment_start,
                annotations=segment_events,
            )
            if json_sidecar:
                if status_cb:
                    status_cb(f"write json (segment {seg_idx + 1}/{len(segments)})")
                _write_json_sidecar(
                    segment_output,
                    source_path=input_path,
                    sampling_rate=seg_fs,
                    channel_labels=recovered_channel_labels,
                    sample_count=seg_waveform.shape[0],
                    start_time=segment_start,
                    events=segment_events,
                    nrv_header=nrv_header,
                )
        logger.info(
            "Converted %s → %s (split into %d segments)",
            input_path.name,
            resolved_output.name,
            len(segments),
        )
        return _segment_output_path(resolved_output, 0, len(segments))

    if resample_to is not None and (mixed_across_channels or mixed_over_time):
        if status_cb:
            status_cb("resample")
        waveform = _read_and_resample_mixed(
            input_path,
            nrv_header,
            channels,
            float(resample_to),
            segment_aware=mixed_over_time,
        )
        fs = float(resample_to)
        if lowcut is not None or highcut is not None:
            if status_cb:
                status_cb("bandpass filter")
            waveform = _bandpass_filter(waveform, float(fs), lowcut, highcut)
        if notch is not None and notch > 0:
            if status_cb:
                status_cb("notch filter")
            waveform = _notch_filter(waveform, float(fs), notch)
        waveform = _apply_ecg_polarity_correction(waveform, ecg_indices)
        adjusted_events = _adjust_events_for_gaps(
            nrv_header.Events, nrv_header.Segments, nrv_header.startDateTime
        )
        if vendor_style:
            adjusted_events = _filter_vendor_events(adjusted_events)
        adjusted_events = normalize_events(adjusted_events)
        adjusted_events = list(adjusted_events) + _sleep_stage_events(nrv_header.SleepScores, nrv_header.startDateTime)
        if status_cb:
            status_cb("write edf")
        write_edf(
            resolved_output, waveform, fs, recovered_channel_labels, patient_meta,
            recording_start=nrv_header.startDateTime, annotations=adjusted_events,
        )
        edf_fs: float | list[float] = fs
        edf_sample_count: int = waveform.shape[0]
    elif mixed_across_channels and resample_to is None:
        # Native multi-rate: read each channel at its own sampling frequency and
        # write them to EDF without any upsampling.
        if status_cb:
            status_cb("read data (multi-rate)")
        ch_data_list, ch_rate_list = _read_channels_native_multirate(
            input_path, nrv_header, channels
        )
        # Apply optional filters per channel at its native rate
        if lowcut is not None or highcut is not None:
            if status_cb:
                status_cb("bandpass filter")
            ch_data_list = [
                _bandpass_filter(d.reshape(-1, 1), r, lowcut, highcut)[:, 0]
                for d, r in zip(ch_data_list, ch_rate_list)
            ]
        if notch is not None and notch > 0:
            if status_cb:
                status_cb("notch filter")
            ch_data_list = [
                _notch_filter(d.reshape(-1, 1), r, notch)[:, 0]
                for d, r in zip(ch_data_list, ch_rate_list)
            ]
        ch_data_list = _apply_ecg_polarity_correction(ch_data_list, ecg_indices)
        adjusted_events = _adjust_events_for_gaps(
            nrv_header.Events, nrv_header.Segments, nrv_header.startDateTime
        )
        if vendor_style:
            adjusted_events = _filter_vendor_events(adjusted_events)
        adjusted_events = normalize_events(adjusted_events)
        adjusted_events = list(adjusted_events) + _sleep_stage_events(nrv_header.SleepScores, nrv_header.startDateTime)
        if status_cb:
            status_cb("write edf")
        write_edf(
            resolved_output, ch_data_list, ch_rate_list, recovered_channel_labels, patient_meta,
            recording_start=nrv_header.startDateTime, annotations=adjusted_events,
        )
        edf_fs = ch_rate_list
        edf_sample_count = max((len(d) for d in ch_data_list), default=0)
    else:
        waveform = read_nervus_data(input_path, nrv_header, channels=channels).T  # samples x channels

        # Apply optional bandpass/highpass/lowpass filter
        if lowcut is not None or highcut is not None:
            if status_cb:
                status_cb("bandpass filter")
            waveform = _bandpass_filter(waveform, float(fs), lowcut, highcut)

        # Apply optional notch filter (powerline removal)
        if notch is not None and notch > 0:
            if status_cb:
                status_cb("notch filter")
            waveform = _notch_filter(waveform, float(fs), notch)

        if resample_to is not None:
            if status_cb:
                status_cb("resample")
            waveform = _resample_waveform(waveform, float(fs), float(resample_to))
            fs = float(resample_to)

        waveform = _apply_ecg_polarity_correction(waveform, ecg_indices)
        adjusted_events = _adjust_events_for_gaps(
            nrv_header.Events, nrv_header.Segments, nrv_header.startDateTime
        )
        if vendor_style:
            adjusted_events = _filter_vendor_events(adjusted_events)
        adjusted_events = normalize_events(adjusted_events)
        adjusted_events = list(adjusted_events) + _sleep_stage_events(nrv_header.SleepScores, nrv_header.startDateTime)
        if status_cb:
            status_cb("write edf")
        write_edf(
            resolved_output, waveform, fs, recovered_channel_labels, patient_meta,
            recording_start=nrv_header.startDateTime, annotations=adjusted_events,
        )
        edf_fs = fs
        edf_sample_count = waveform.shape[0]

    if json_sidecar:
        if status_cb:
            status_cb("write json")
        _write_json_sidecar(
            resolved_output,
            source_path=input_path,
            sampling_rate=edf_fs if isinstance(edf_fs, float) else edf_fs[0],
            channel_labels=recovered_channel_labels,
            sample_count=edf_sample_count,
            start_time=nrv_header.startDateTime,
            events=adjusted_events,
            nrv_header=nrv_header,
        )
    logger.info(
        "Converted %s → %s (%d channels @ %.2f Hz)",
        input_path.name,
        resolved_output.name,
        len(recovered_channel_labels),
        fs,
    )
    return resolved_output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "ui", False):
        if not rich_available():
            print(
                "UI mode requires 'rich'.\n\n"
                "Install it (pick one):\n"
                "- uv run --with rich nicolet-e2edf --ui\n"
                "- uv pip install -p .venv rich && nicolet-e2edf --ui\n"
                "- uv pip install -p .venv '.[tui]' && nicolet-e2edf --ui\n"
            )
            return 1
        args = run_rich_wizard(title="nicolet-e2edf")

    if not args.input_paths:
        parser.error("--in is required (or run the interactive UI with --ui)")
    if not args.output_dir:
        parser.error("--out is required (or run the interactive UI with --ui)")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    input_paths = [Path(raw).expanduser() for raw in args.input_paths]
    output_dir = Path(args.output_dir).expanduser()
    patient_rules = _load_patient_rules(args.patient_json)

    try:
        inputs_with_roots = _discover_inputs_with_roots(input_paths, args.glob)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not inputs_with_roots:
        logger.error("No input files matched the supplied path/pattern")
        return 1

    candidate_outputs: dict[tuple[Path, Path | None], Path] = {}
    collisions: dict[Path, list[tuple[Path, Path | None]]] = {}
    for source_path, source_root in inputs_with_roots:
        candidate = _candidate_output_path(source_path, output_dir, source_root)
        key = (source_path, source_root)
        candidate_outputs[key] = candidate
        collisions.setdefault(candidate, []).append(key)

    resolved_outputs: dict[tuple[Path, Path | None], Path] = {}
    for candidate, sources in collisions.items():
        if len(sources) == 1:
            resolved_outputs[sources[0]] = candidate
            continue
        for source_path, source_root in sources:
            digest = hashlib.sha1(
                str(source_path).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:8].upper()
            resolved_outputs[(source_path, source_root)] = candidate.with_name(
                f"{candidate.stem}_{digest}{candidate.suffix}"
            )

    if getattr(args, "ui", False):
        tui_inputs = [
            (source_path, source_root, resolved_outputs[(source_path, source_root)])
            for source_path, source_root in inputs_with_roots
        ]

        def _convert_one(*, source_path: Path, output_path: Path, input_root: Path | None, status_cb) -> None:
            convert_file(
                source_path,
                output_dir,
                patient_rules,
                output_path=output_path,
                input_root=input_root,
                json_sidecar=args.json_sidecar,
                resample_to=args.resample_to,
                lowcut=getattr(args, "lowcut", None),
                highcut=getattr(args, "highcut", None),
                notch=getattr(args, "notch", None),
                split_by_segment=bool(getattr(args, "split_by_segment", False)),
                vendor_style=bool(getattr(args, "vendor_style", False)),
                status_cb=status_cb,
            )

        return run_tui(
            inputs=tui_inputs,
            options=TuiOptions(
                json_sidecar=bool(getattr(args, "json_sidecar", False)),
                resample_to=getattr(args, "resample_to", None),
                lowcut=getattr(args, "lowcut", None),
                highcut=getattr(args, "highcut", None),
                notch=getattr(args, "notch", None),
                verbose=bool(getattr(args, "verbose", False)),
            ),
            convert_one=_convert_one,
            title="nicolet-e2edf",
        )

    success = 0
    for source_path, source_root in inputs_with_roots:
        try:
            convert_file(
                source_path,
                output_dir,
                patient_rules,
                output_path=resolved_outputs[(source_path, source_root)],
                input_root=source_root,
                json_sidecar=args.json_sidecar,
                resample_to=args.resample_to,
                lowcut=args.lowcut,
                highcut=args.highcut,
                notch=args.notch,
                split_by_segment=bool(getattr(args, "split_by_segment", False)),
                vendor_style=bool(getattr(args, "vendor_style", False)),
            )
            success += 1
        except Exception as exc:  # pragma: no cover - logged for CLI feedback
            logger.error("Failed to convert %s: %s", source_path, exc)
    if success == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
