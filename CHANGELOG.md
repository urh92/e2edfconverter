# Changelog

## Unreleased

-

## 0.2.9 (2026-03-03)

- Fix duplicate channel-label collisions in legacy `.eeg` recordings by disambiguating repeated labels with references when available (e.g. `Fp1-Ref`, `Fp1-AV`).
- Add deterministic fallback suffixing (e.g. `_2`) when duplicates remain after reference disambiguation.
- Improve channel-type categorization for suffixed/disambiguated labels (e.g. `Fp1-AV`, `Photic_2`).
- Note: some legacy `.eeg` files legitimately contain repeated electrode names because multiple reference sets are stored in one recording (for example both `Ref` and `AV` channels).

## 0.2.8 (2026-03-03)

- Fix critical header-parse bug resulting in hangup on some multichannel files by hardening UNKNOWN montage-catalog parsing.
- Add guardrails for oversized UNKNOWN blobs/token streams and lightweight caching in catalog title/channel checks.

## 0.2.7 (2026-02-26)

- SSE 64-channel numeric-ID recovery and README updates.
- Fix case-sensitive REF reference inference.

## 0.2.6 (2026-02-23)

- Faster conversion: improved channel-window reads and montage/header parsing.

## 0.2.5 (2026-02-22)

- Added new channel-label recovery for numeric-ID recordings by adding hidden montage catalog and fixed-table heuristics.
- Fix auxiliary montage conflict handling when catalog rows are interleaved during scoring-based ordering.
- Prevent derived/bipolar labels (e.g. `Fp2-av`) from being used as direct channel names during hidden-catalog recovery.
- Parse `InputInfo` / `InputSettingsInfo` from source headers for internal diagnostics. Perhaps this is useful later ...
- Remove dead code and stale no-op variables in conversion and legacy parsing paths.
- Add regression tests for interleaved auxiliary catalog conflicts and bipolar-label rejection in unknown catalogs.

## 0.2.4 (2026-02-20)

- Enforce integer-Hz EDF output sampling rates to prevent silent timing/sample drift.
- Reject fractional `--resample-to` values at CLI parsing time.
- Fail early with a clear error when source rate is fractional and no resampling is requested.
- Add regression tests for fractional-rate rejection paths.

## 0.2.3 (2026-02-18)

- Improve robustness of event parsing when metadata is split across multiple sections.
- Improve event type resolution from metadata, reducing unknown event labels.
- Improve label text handling for non-ASCII Latin characters.
- Improve annotation parity with proprietary exports.
- Thanks to Sampsa for providing test files used in this iteration!!!

## 0.2.2 (2026-01-10)

- Resampling: `--resample-to` uses `scipy.signal.resample_poly` (polyphase FIR) and requires scipy.

## 0.2.1 (2026-01-08)

- Add `--split-by-segment` and `--vendor-style`.
- Improve UTF-16 label scanning and event label handling.
- Docs updates (including `.eeg` support status).

## 0.2.0 (2026-01-07)

- Legacy `.eeg` support (experimental).
- Mixed-rate handling via `--resample-to` (including segment-aware resampling).
- Better parsing for segments, channel on/off handling, and EEG offset support.
- EDF+ writer improvements + stricter validation (PyEDFlib).
- JSON sidecar improvements.

## 0.1.1 (2026-01-05)

- Packaging and CLI polish (quick start, CLI options).

## 0.1.0 (2025-12)

- Initial `.e` → EDF converter with EDF+ annotations.
- Optional TUI, filtering, and JSON sidecar support.
