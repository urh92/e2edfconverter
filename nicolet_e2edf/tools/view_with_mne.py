from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nicolet_e2edf.nicolet.data import read_nervus_data
from nicolet_e2edf.nicolet.header import read_nervus_header

if TYPE_CHECKING:  # pragma: no cover
    import mne


def _clean_label(label: str) -> str:
    text = label.split("\x00", 1)[0].strip()
    text = text.encode("ascii", "ignore").decode("ascii")
    return text or "Ch"


def _build_annotations(nrv_header) -> mne.Annotations | None:
    import mne

    if not nrv_header.Events or not nrv_header.startDateTime:
        return None
    onsets = []
    durations = []
    descriptions = []
    start = nrv_header.startDateTime.replace(tzinfo=timezone.utc)
    for event in nrv_header.Events:
        evt_time = event.date.replace(tzinfo=timezone.utc)
        onset = (evt_time - start).total_seconds()
        if onset < 0:
            continue
        onsets.append(onset)
        durations.append(float(event.duration) if event.duration else 0.0)
        descriptions.append(event.IDStr or event.label or "Event")
    if not onsets:
        return None
    return mne.Annotations(onsets, durations, descriptions)


def launch_viewer(path: Path, start: float, duration: float, channels: Iterable[int] | None, notch: float) -> None:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install 'mne' to use this viewer.\n"
            "Recommended: uv run --with mne ...\n"
            "Or install manually: uv sync --with viewer"
        ) from exc

    public_header, nrv_header = read_nervus_header(path)
    fs = public_header["Fs"] or nrv_header.targetSamplingRate
    if not fs:
        raise RuntimeError("Unable to determine sampling rate from header")

    if channels is None:
        chan_idx = nrv_header.matchingChannels or list(range(1, len(nrv_header.TSInfo) + 1))
    else:
        chan_idx = list(channels)
    zero_based = [idx - 1 for idx in chan_idx]

    begsample = int(start * fs) + 1
    endsample = begsample + int(duration * fs) if duration > 0 else None

    data = read_nervus_data(path, nrv_header, channels=chan_idx, begsample=begsample, endsample=endsample)
    data = data.astype(np.float64) * 1e-6  # µV -> V

    labels = []
    for idx in zero_based:
        try:
            labels.append(_clean_label(nrv_header.Segments[0].chName[idx]))
        except (IndexError, AttributeError):
            labels.append(f"Ch{idx + 1}")

    info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=["eeg"] * len(labels))
    raw = mne.io.RawArray(data, info, verbose=False)

    annotations = _build_annotations(nrv_header)
    if annotations is not None:
        raw.set_annotations(annotations)

    if notch and notch > 0:
        raw.notch_filter(float(notch), verbose=False)
    raw.plot(start=start, duration=duration if duration > 0 else 10.0, block=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch MNE viewer for a Nicolet .e file")
    parser.add_argument("path", type=Path, help="Path to Nicolet .e file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration to load (seconds; 0 = to end)")
    parser.add_argument(
        "--notch",
        type=float,
        default=50.0,
        help="Notch filter frequency in Hz (default: %(default)s; set 0 to disable)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        help="1-based channel indices to view (default: dominant sampling-rate channels)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    launch_viewer(args.path, args.start, args.duration, args.channels, args.notch)


if __name__ == "__main__":  # pragma: no cover
    main()
