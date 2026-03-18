from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet.data import read_nervus_data
from nicolet_e2edf.nicolet.edf_writer import write_edf
from nicolet_e2edf.nicolet.types import MainIndexEntry, NervusHeader, SegmentInfo, StaticPacket


def _build_synthetic_header(path: Path, samples: np.ndarray) -> NervusHeader:
    header = NervusHeader(filename=path)
    header.StaticPackets = [StaticPacket(tag="0", index=1, IDStr="0")]
    header.MainIndex = [
        MainIndexEntry(sectionIdx=1, offset=0, blockL=samples.nbytes, sectionL=samples.nbytes)
    ]
    segment = SegmentInfo(
        dateOLE=0.0,
        date=datetime.now(timezone.utc),
        duration=1.0,
        chName=["C1"],
        refName=["REF"],
        samplingRate=np.array([samples.size], dtype=float),
        scale=np.array([2.0], dtype=float),
        sampleCount=np.array([samples.size], dtype=int),
    )
    header.Segments = [segment]
    header.matchingChannels = [1]
    return header


def test_data_reader_subrange_matches_slice(tmp_path: Path) -> None:
    raw = np.arange(1, 11, dtype="<i2")
    recording = tmp_path / "mock.e"
    recording.write_bytes(raw.tobytes())
    header = _build_synthetic_header(recording, raw)

    full = read_nervus_data(recording, header)
    assert full.shape == (1, raw.size)
    np.testing.assert_array_equal(full, raw[np.newaxis, :] * 2.0)

    subset = read_nervus_data(recording, header, begsample=3, endsample=6)
    np.testing.assert_array_equal(subset, full[:, 2:6])


def test_write_edf_rejects_fractional_sampling_rate(tmp_path: Path) -> None:
    out = tmp_path / "fractional.edf"
    data = np.zeros((100, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="integer sampling rate"):
        write_edf(out, data, 128.5, ["C3"])
