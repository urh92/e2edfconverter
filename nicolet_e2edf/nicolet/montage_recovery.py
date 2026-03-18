"""Planning/scoring helpers for montage-based channel-label recovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

AUXILIARY_MONTAGE_SOURCES = frozenset(
    {"aux_av_table", "supplemental_generic", "unknown_montage_catalog"}
)
DIRECT_ID_NAME_SOURCES = frozenset({"derivation_fixed_table", "unknown_montage_catalog"})

_NUMERIC_REF_DERIVATION_RE = re.compile(r"^\d+\s*-\s*REF$", re.IGNORECASE)
_NON_WORD_SPLIT = re.compile(r"[^A-Z0-9]+")


@dataclass(slots=True)
class MontageRecoveryPlan:
    """Partitioned montage rows and source diagnostics for recovery."""

    primary_entries: list[dict[str, object]]
    auxiliary_entries: list[dict[str, object]]
    has_unknown_catalog: bool
    source_counts: dict[str, int]
    source_scores: dict[str, dict[str, int]]
    catalog_scores: dict[str, dict[str, int]]


def _clean_token(value: object) -> str:
    return str(value or "").split("\x00", 1)[0].strip()


def _match_key(value: object) -> str:
    token = _clean_token(value).upper()
    return _NON_WORD_SPLIT.sub("", token)


def _is_numeric_signal_id(value: object) -> bool:
    return _clean_token(value).isdigit()


def _is_numeric_like_label(value: object) -> bool:
    return bool(re.fullmatch(r"\d+", _clean_token(value)))


def should_skip_main_direct_numeric_ref_fallback(
    source: object,
    derivation_name: object,
    *,
    has_unknown_catalog: bool,
) -> bool:
    """Return True when ``N-Ref`` rows should not become final labels."""

    if not has_unknown_catalog:
        return False
    if str(source or "") != "derivation_main_table":
        return False
    return bool(_NUMERIC_REF_DERIVATION_RE.fullmatch(_clean_token(derivation_name)))


def _score_montage_sources(
    montage_entries: list[dict[str, object]],
    target_signal_ids: set[str],
) -> dict[str, dict[str, int]]:
    scores: dict[str, dict[str, int]] = {}
    for entry in montage_entries:
        source = str(entry.get("source", ""))
        score = scores.setdefault(
            source,
            {
                "rows": 0,
                "target_rows": 0,
                "target_numeric_names": 0,
                "target_named_rows": 0,
                "named_rows": 0,
            },
        )
        score["rows"] += 1

        signal_name_1 = _clean_token(entry.get("signalName1", ""))
        derivation_name = _clean_token(entry.get("derivationName", ""))
        if derivation_name and not _is_numeric_like_label(derivation_name):
            score["named_rows"] += 1

        if signal_name_1 and _match_key(signal_name_1) in target_signal_ids:
            score["target_rows"] += 1
            if derivation_name and not _is_numeric_like_label(derivation_name):
                score["target_named_rows"] += 1
            elif derivation_name:
                score["target_numeric_names"] += 1

    return scores


def _is_ref_like_label(label: str) -> bool:
    upper = _clean_token(label).upper()
    if not upper:
        return False
    if upper in {"AV", "REF", "REFERENCE"}:
        return True
    return bool(_NUMERIC_REF_DERIVATION_RE.fullmatch(upper))


def _is_simple_contact_label(label: str) -> bool:
    cleaned = _clean_token(label)
    if not cleaned or _is_numeric_like_label(cleaned) or _is_ref_like_label(cleaned):
        return False
    # Bipolar/derived names are useful for diagnostics but are worse than
    # direct contact/electrode names for channel relabeling.
    if any(sep in cleaned for sep in ("-", "/", "\\", "+", ":")):
        return False
    return True


def _score_unknown_catalog_groups(
    unknown_entries: list[dict[str, object]],
    target_signal_ids: set[str],
) -> dict[str, dict[str, int]]:
    scores: dict[str, dict[str, int]] = {}
    for entry in unknown_entries:
        catalog = str(entry.get("montageName", "") or "").strip() or "<unnamed>"
        score = scores.setdefault(
            catalog,
            {
                "rows": 0,
                "target_rows": 0,
                "target_named_rows": 0,
                "target_simple_named_rows": 0,
                "target_numeric_names": 0,
                "target_ref_like_rows": 0,
                "target_bipolar_rows": 0,
            },
        )
        score["rows"] += 1
        signal_name_1 = _clean_token(entry.get("signalName1", ""))
        derivation_name = _clean_token(entry.get("derivationName", ""))
        if not signal_name_1 or _match_key(signal_name_1) not in target_signal_ids:
            continue
        score["target_rows"] += 1
        if not derivation_name:
            continue
        if _is_numeric_like_label(derivation_name):
            score["target_numeric_names"] += 1
            continue
        score["target_named_rows"] += 1
        if _is_ref_like_label(derivation_name):
            score["target_ref_like_rows"] += 1
        if any(sep in derivation_name for sep in ("-", "/", "\\", "+", ":")):
            score["target_bipolar_rows"] += 1
        if _is_simple_contact_label(derivation_name):
            score["target_simple_named_rows"] += 1
    return scores


def _unknown_catalog_is_usable(score: dict[str, int]) -> bool:
    """Return True when a hidden catalog is useful for direct channel naming."""

    target_rows = int(score.get("target_rows", 0))
    simple_rows = int(score.get("target_simple_named_rows", 0))
    named_rows = int(score.get("target_named_rows", 0))
    ref_rows = int(score.get("target_ref_like_rows", 0))
    if target_rows <= 0 or named_rows <= 0:
        return False
    if simple_rows <= 0:
        return False
    # Require at least a modest amount of useful coverage or a strong ratio.
    return simple_rows >= 8 or (target_rows > 0 and simple_rows * 2 >= target_rows and ref_rows == 0)


def _aux_source_rank(source: str) -> int:
    # Lower is better. Keep `unknown_montage_catalog` ahead of the looser token
    # scanners because it gives direct `ID -> name` mappings.
    if source == "unknown_montage_catalog":
        return 0
    if source == "aux_av_table":
        return 1
    if source == "supplemental_generic":
        return 2
    return 9


def build_montage_recovery_plan(
    montage_entries: list[dict[str, object]] | None,
    channel_labels: list[str],
) -> MontageRecoveryPlan:
    """Partition and order montage rows used during label recovery.

    Primary rows are preserved in file order because they reflect the current
    montage's explicit derivation rows. Auxiliary sources are sorted by a small
    scoring heuristic against the active numeric channel IDs so richer direct
    catalogs (e.g. hidden `128 kanaler`) are considered before generic tables.
    """

    entries = list(montage_entries or [])
    numeric_targets = {
        _match_key(label)
        for label in channel_labels
        if _is_numeric_signal_id(label)
    }
    source_counts = dict(Counter(str(entry.get("source", "")) for entry in entries))
    source_scores = _score_montage_sources(entries, numeric_targets)

    primary_entries: list[dict[str, object]] = []
    auxiliary_entries: list[dict[str, object]] = []
    for entry in entries:
        source = str(entry.get("source", ""))
        if source in AUXILIARY_MONTAGE_SOURCES:
            auxiliary_entries.append(entry)
        else:
            primary_entries.append(entry)

    unknown_entries = [
        entry for entry in auxiliary_entries if str(entry.get("source", "")) == "unknown_montage_catalog"
    ]
    unknown_catalog_scores = _score_unknown_catalog_groups(unknown_entries, numeric_targets)
    usable_unknown_catalogs = {
        catalog for catalog, score in unknown_catalog_scores.items() if _unknown_catalog_is_usable(score)
    }
    if unknown_entries:
        auxiliary_entries = [
            entry
            for entry in auxiliary_entries
            if str(entry.get("source", "")) != "unknown_montage_catalog"
            or (str(entry.get("montageName", "") or "").strip() or "<unnamed>") in usable_unknown_catalogs
        ]

    # Stable sort: score by catalog/source quality, preserve original order inside ties.
    aux_index = {id(entry): idx for idx, entry in enumerate(auxiliary_entries)}

    def _aux_sort_key(entry: dict[str, object]) -> tuple[int, ...]:
        source = str(entry.get("source", ""))
        if source == "unknown_montage_catalog":
            catalog = str(entry.get("montageName", "") or "").strip() or "<unnamed>"
            catalog_score = unknown_catalog_scores.get(catalog, {})
            # Prefer hidden catalogs that best name the active channels with
            # simple non-numeric labels, and de-prioritize AV/numeric/bipolar
            # catalogs when a better option exists.
            return (
                -catalog_score.get("target_simple_named_rows", 0),
                -catalog_score.get("target_named_rows", 0),
                -catalog_score.get("target_rows", 0),
                catalog_score.get("target_ref_like_rows", 0),
                catalog_score.get("target_numeric_names", 0),
                catalog_score.get("target_bipolar_rows", 0),
                _aux_source_rank(source),
                aux_index[id(entry)],
            )
        return (
            -source_scores.get(source, {}).get("target_named_rows", 0),
            -source_scores.get(source, {}).get("target_rows", 0),
            -source_scores.get(source, {}).get("named_rows", 0),
            _aux_source_rank(source),
            aux_index[id(entry)],
        )

    auxiliary_entries.sort(
        key=_aux_sort_key
    )

    return MontageRecoveryPlan(
        primary_entries=primary_entries,
        auxiliary_entries=auxiliary_entries,
        has_unknown_catalog=bool(usable_unknown_catalogs),
        source_counts=source_counts,
        source_scores=source_scores,
        catalog_scores=unknown_catalog_scores,
    )
