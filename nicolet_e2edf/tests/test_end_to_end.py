from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet import cli
from nicolet_e2edf.nicolet.header import normalize_events
from nicolet_e2edf.nicolet.types import EventItem, NervusHeader, SegmentInfo


def test_categorize_channel_treats_pg1_pg2_as_eeg() -> None:
    assert cli._categorize_channel("Pg1") == "EEG"
    assert cli._categorize_channel("PG2") == "EEG"
    assert cli._categorize_channel("Fp1-AV") == "EEG"
    assert cli._categorize_channel("Photic_2") == "Stimulus"


def test_channel_labels_disambiguates_duplicate_names_with_reference() -> None:
    fake_header = NervusHeader(filename=Path("/tmp/legacy.eeg"))
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["Fp1", "Fp1", "Photic", "Photic"],
            refName=["Ref", "AV", "Ref", "Ref"],
            samplingRate=np.array([128.0, 128.0, 128.0, 128.0]),
            scale=np.ones(4),
            sampleCount=np.array([4, 4, 4, 4]),
        )
    ]

    labels = cli._channel_labels(fake_header, [1, 2, 3, 4])

    assert labels == ["Fp1-Ref", "Fp1-AV", "Photic", "Photic_2"]


def test_should_attempt_numeric_montage_recovery_with_one_named_other_tail() -> None:
    labels = [str(i) for i in range(1, 65)] + ["NpsykPC"]
    assert cli._should_attempt_numeric_montage_recovery(labels) is True


def test_should_attempt_numeric_montage_recovery_with_numeric_ref_placeholders() -> None:
    labels = [f"{i}-Ref" for i in range(1, 65)]
    assert cli._should_attempt_numeric_montage_recovery(labels) is True


def test_sse_fixed_64_numeric_id_mapping_recovers_line_layout() -> None:
    fake_header = NervusHeader(filename=Path("/tmp/SSE/Arkiv/Patient.e"))
    labels = [str(i) for i in range(1, 65)]

    recovered = cli._recover_channel_labels_from_montage(fake_header, labels)

    assert len(recovered) == 64
    assert recovered[-1] == "EKG"
    assert recovered[:6] == ["FP1", "FP2", "AF7", "AF8", "AF3", "AF4"]
    assert recovered[32:36] == ["TP9", "TP10", "TP7", "TP8"]
    assert recovered[-8:] == ["FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ", "EKG"]


def test_sse_fixed_64_numeric_id_mapping_is_path_gated() -> None:
    fake_header = NervusHeader(filename=Path("/tmp/other_site/Patient.e"))
    labels = [str(i) for i in range(1, 65)]

    recovered = cli._recover_channel_labels_from_montage(fake_header, labels)

    assert recovered == labels


def test_convert_to_edf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2021, 5, 5, 8, 30, 1),
            duration=2.0,
            user="user",
            GUID="{GUID}",
            label="TestEvent",
            IDStr="TestEvent",
            annotation="note",
        )
    ]

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir)])
    assert exit_code == 0

    edf_path = output_dir / "case.edf"
    assert edf_path.exists()

    content = edf_path.read_bytes()
    header = content[:256 + 256 * 2]  # base + two signals
    n_signals = int(header[252:256].decode("ascii").strip())
    assert n_signals == 2

    label_section = header[256 : 256 + 32]
    assert label_section[:16].decode("ascii").strip() == "C3"
    assert label_section[16:32].decode("ascii").strip() == "EDF Annotations"

    samples_offset = 256
    samples_offset += 16 * n_signals  # labels
    samples_offset += 80 * n_signals  # transducer
    samples_offset += 8 * n_signals  # physical dimension
    samples_offset += 8 * n_signals  # physical min
    samples_offset += 8 * n_signals  # physical max
    samples_offset += 8 * n_signals  # digital min
    samples_offset += 8 * n_signals  # digital max
    samples_offset += 80 * n_signals  # prefilter
    samples_section = header[samples_offset : samples_offset + 8 * n_signals]
    samples_counts = [
        int(samples_section[i * 8 : (i + 1) * 8].decode("ascii").strip()) for i in range(n_signals)
    ]
    
    # With 1-second data records, samples_per_record = sampling frequency
    # The actual data (4 samples) is padded to fill the 1-second record
    assert samples_counts[0] == 128  # samples per 1-second record at 128 Hz

    # Verify annotation signal has enough samples for TAL data
    # The annotation samples are sized to hold all events for the worst-case record
    assert samples_counts[1] >= 8  # minimum annotation samples


def test_resample_and_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("scipy", reason="scipy required for resampling")
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2021, 5, 5, 8, 30, 1),
            duration=2.0,
            user="user",
            GUID="{GUID}",
            label="TestEvent",
            IDStr="TestEvent",
            annotation="note",
        )
    ]

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[0, 10, 20, 30]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        ["--in", str(recording), "--out", str(output_dir), "--json-sidecar", "--resample-to", "64"]
    )
    assert exit_code == 0

    edf_path = output_dir / "case.edf"
    sidecar_path = output_dir / "case.json"
    assert edf_path.exists()
    assert sidecar_path.exists()

    content = edf_path.read_bytes()
    header = content[:256 + 256 * 2]
    samples_offset = 256
    n_signals = int(header[252:256].decode("ascii").strip())
    samples_offset += 16 * n_signals
    samples_offset += 80 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 80 * n_signals
    samples_section = header[samples_offset : samples_offset + 8 * n_signals]
    samples_counts = [
        int(samples_section[i * 8 : (i + 1) * 8].decode("ascii").strip()) for i in range(n_signals)
    ]
    
    # With 1-second data records, samples_per_record = resampled frequency
    # The actual data (2 samples after resampling to 64 Hz) is padded to fill 1-second record
    assert samples_counts[0] == 64  # samples per 1-second record at 64 Hz

    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["sampling_rate_hz"] == 64
    # sample_count in sidecar is the ACTUAL sample count, not padded
    assert sidecar["sample_count"] == 2
    # Events with annotation text go to "annotations" list, not "events"
    assert sidecar["annotations"][0]["onset_seconds"] == 1.0


def test_prune_channel_label_is_suppressed_in_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyedflib = pytest.importorskip("pyedflib", reason="pyedflib required for EDF annotation checks")

    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 100.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=2.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([100.0]),
            scale=np.ones(1),
            sampleCount=np.array([200]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2021, 5, 5, 8, 30, 1),
            duration=1.5,
            user="user",
            GUID="{GUID}",
            label="Fp1-SFp1",
            IDStr="Prune",
            annotation=None,
        )
    ]

    def _fake_read_header(path: Path):
        return {"Fs": 100.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.arange(200, dtype=np.float32).reshape(1, 200)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar_path = output_dir / "case.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["events"] == [
        {
            "onset_seconds": 1.0,
            "duration_seconds": 1.5,
            "type": "Prune",
            "label": None,
        }
    ]

    reader = pyedflib.EdfReader(str(output_dir / "case.edf"))
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()

    descriptions = list(descriptions)
    assert "Prune" in descriptions
    assert "Fp1-SFp1" not in descriptions


def test_normalize_events_preserves_raw_prune_label_but_clears_export_label() -> None:
    events = normalize_events([
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2021, 5, 5, 8, 30, 1),
            duration=1.5,
            user="user",
            GUID="{GUID}",
            label="Fp1-SFp1",
            IDStr="Prune",
            annotation=None,
        )
    ])

    assert len(events) == 1
    assert events[0].label is None
    assert events[0].rawLabel == "Fp1-SFp1"


def test_montage_mapping_recovers_numeric_channel_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3, 4]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["37", "13", "38", "14"],
            refName=["REF", "REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0, 128.0]),
            scale=np.ones(4),
            sampleCount=np.array([4, 4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "FP1-av", "signalName1": "37", "signalName2": "AV26"},
        {"derivationName": "FP2-av", "signalName1": "13", "signalName2": "AV26"},
        {"derivationName": "F3-av", "signalName1": "38", "signalName2": "AV26"},
        {"derivationName": "F4-av", "signalName1": "14", "signalName2": "AV26"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42], [13, 23, 33, 43]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar_path = output_dir / "case.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())

    assert sidecar["eeg_channel_count"] == 4
    assert [ch["name"] for ch in sidecar["channels"]] == ["FP1", "FP2", "F3", "F4"]
    assert all("original_name" not in ch for ch in sidecar["channels"])
    assert all("name_source" not in ch for ch in sidecar["channels"])
    assert "channel_name_recovery_applied" not in sidecar

    edf_path = output_dir / "case.edf"
    content = edf_path.read_bytes()
    header = content[:256 + 256 * 5]  # 4 signals + annotation
    label_section = header[256 : 256 + 16 * 5]
    labels = [
        label_section[i * 16 : (i + 1) * 16].decode("ascii").strip()
        for i in range(5)
    ]
    assert labels[:4] == ["FP1", "FP2", "F3", "F4"]


def test_montage_mapping_skipped_for_non_numeric_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["VST 01", "VST 02"],
            refName=["REF", "REF"],
            samplingRate=np.array([128.0, 128.0]),
            scale=np.ones(2),
            sampleCount=np.array([4, 4]),
        )
    ]
    # Even if derivation names look like scalp channels, recovery must not run
    # unless the source channel labels are numeric-style and EEG count is zero.
    fake_header.MontageInfo = [
        {"derivationName": "FP1-av", "signalName1": "VST 01", "signalName2": "AV26"},
        {"derivationName": "FP2-av", "signalName1": "VST 02", "signalName2": "AV26"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40], [11, 21, 31, 41]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar_path = output_dir / "case.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert "channel_name_recovery_applied" not in sidecar
    assert [ch["name"] for ch in sidecar["channels"]] == ["VST 01", "VST 02"]


def test_montage_mapping_recovers_from_signalname2_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["22", "23", "27"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "T10-P10", "signalName1": "22", "signalName2": "23"},
        {"derivationName": "CZ-PZ", "signalName1": "26", "signalName2": "27"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert "channel_name_recovery_applied" not in sidecar
    names = [ch["name"] for ch in sidecar["channels"]]
    assert names == ["T10", "P10", "PZ"]


def test_montage_mapping_recovers_when_most_channels_are_numeric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["37", "13", "PZ"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "FP1-av", "signalName1": "37", "signalName2": "AV64"},
        {"derivationName": "FP2-av", "signalName1": "13", "signalName2": "AV64"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    names = [ch["name"] for ch in sidecar["channels"]]
    assert names == ["FP1", "FP2", "PZ"]


def test_montage_mapping_handles_parenthetical_alias_in_derivation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["25"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "T7(T3)-av", "signalName1": "25", "signalName2": "AV64"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    names = [ch["name"] for ch in sidecar["channels"]]
    assert names == ["T7"]


def test_montage_mapping_recovers_non_eeg_single_derivation_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["68", "61"],
            refName=["REF", "REF"],
            samplingRate=np.array([128.0, 128.0]),
            scale=np.ones(2),
            sampleCount=np.array([4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "EKG", "signalName1": "68", "signalName2": ""},
        {"derivationName": "PZ-av", "signalName1": "61", "signalName2": "AV64"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40], [11, 21, 31, 41]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["EKG", "PZ"]


def test_montage_mapping_aux_rows_fill_missing_without_overriding_primary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["44", "62", "EKG"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        # Primary row should win for 44.
        {"derivationName": "01-av", "signalName1": "44", "signalName2": "AV26"},
        # Auxiliary rows should only fill missing IDs (62) and not override 44.
        {"derivationName": "P6-av", "signalName1": "44", "signalName2": "AV64", "source": "aux_av_table"},
        {"derivationName": "POZ-av", "signalName1": "62", "signalName2": "AV64", "source": "aux_av_table"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    names = [ch["name"] for ch in sidecar["channels"]]
    assert names == ["O1", "POZ", "EKG"]


def test_montage_mapping_recovers_from_derivation_fixed_table_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["1", "69", "82"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "VTP1", "signalName1": "1", "signalName2": "", "source": "derivation_fixed_table"},
        {"derivationName": "F3", "signalName1": "69", "signalName2": "", "source": "derivation_fixed_table"},
        {"derivationName": "CZ", "signalName1": "82", "signalName2": "", "source": "derivation_fixed_table"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["VTP1", "F3", "CZ"]


def test_montage_mapping_recovers_from_derivation_main_table_direct_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["1", "2", "69"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "VTP1", "signalName1": "1", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "VTP2", "signalName1": "2", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "F3", "signalName1": "69", "signalName2": "", "source": "derivation_main_table"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["VTP1", "VTP2", "F3"]


def test_montage_mapping_unknown_catalog_fills_numeric_ref_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["1", "2", "65"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "1-Ref", "signalName1": "1", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "2-Ref", "signalName1": "2", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "FPM1", "signalName1": "65", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "FP1", "signalName1": "1", "signalName2": "", "source": "unknown_montage_catalog"},
        {"derivationName": "FP2", "signalName1": "2", "signalName2": "", "source": "unknown_montage_catalog"},
        # Aux catalog must not override an explicit primary name for 65.
        {"derivationName": "FC5", "signalName1": "65", "signalName2": "", "source": "unknown_montage_catalog"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["FP1", "FP2", "FPM1"]


def test_montage_mapping_prefers_best_unknown_catalog_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["1", "2", "3"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "1-Ref", "signalName1": "1", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "2-Ref", "signalName1": "2", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "3-Ref", "signalName1": "3", "signalName2": "", "source": "derivation_main_table"},
        # A weak hidden catalog (numeric/ref-like names) should not win.
        {"montageName": "AVERAGE", "derivationName": "1", "signalName1": "1", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "AVERAGE", "derivationName": "2", "signalName1": "2", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "AVERAGE", "derivationName": "AV", "signalName1": "3", "signalName2": "", "source": "unknown_montage_catalog"},
        # A richer hidden catalog with direct contact names should win.
        {"montageName": "LOCALCUSTOM", "derivationName": "VTP1", "signalName1": "1", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "LOCALCUSTOM", "derivationName": "VTP2", "signalName1": "2", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "LOCALCUSTOM", "derivationName": "F3", "signalName1": "3", "signalName2": "", "source": "unknown_montage_catalog"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["VTP1", "VTP2", "F3"]


def test_montage_mapping_interleaved_aux_rows_keep_catalog_conflicts_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2, 3]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["1", "2", "3"],
            refName=["REF", "REF", "REF"],
            samplingRate=np.array([128.0, 128.0, 128.0]),
            scale=np.ones(3),
            sampleCount=np.array([4, 4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "1-Ref", "signalName1": "1", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "2-Ref", "signalName1": "2", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "3-Ref", "signalName1": "3", "signalName2": "", "source": "derivation_main_table"},
        # Catalog A and B are intentionally interleaved to mimic real-world
        # score-sorted auxiliary ordering.
        {"montageName": "CATA", "derivationName": "F3", "signalName1": "1", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "CATB", "derivationName": "P3", "signalName1": "2", "signalName2": "", "source": "unknown_montage_catalog"},
        # Conflict for signal 1 inside CATA: this should invalidate CATA's map
        # for that signal instead of silently choosing first-seen value.
        {"montageName": "CATA", "derivationName": "CZ", "signalName1": "1", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "CATB", "derivationName": "P4", "signalName1": "3", "signalName2": "", "source": "unknown_montage_catalog"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["1", "P3", "P4"]


def test_montage_mapping_ignores_bipolar_unknown_catalog_direct_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1, 2]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["1", "2"],
            refName=["REF", "REF"],
            samplingRate=np.array([128.0, 128.0]),
            scale=np.ones(2),
            sampleCount=np.array([4, 4]),
        )
    ]
    fake_header.MontageInfo = [
        {"derivationName": "1-Ref", "signalName1": "1", "signalName2": "", "source": "derivation_main_table"},
        {"derivationName": "2-Ref", "signalName1": "2", "signalName2": "", "source": "derivation_main_table"},
        # Derived labels should not be used as direct channel names.
        {"montageName": "LOCAL", "derivationName": "Fp2-av", "signalName1": "1", "signalName2": "", "source": "unknown_montage_catalog"},
        {"montageName": "LOCAL", "derivationName": "F3", "signalName1": "2", "signalName2": "", "source": "unknown_montage_catalog"},
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array(
            [[10, 20, 30, 40], [11, 21, 31, 41]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir), "--json-sidecar"])
    assert exit_code == 0

    sidecar = json.loads((output_dir / "case.json").read_text())
    assert [ch["name"] for ch in sidecar["channels"]] == ["1", "F3"]


def test_directory_input_preserves_subfolders_and_avoids_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    input_root = tmp_path / "inputs_nested"
    (input_root / "A").mkdir(parents=True)
    (input_root / "B").mkdir(parents=True)
    (input_root / "A" / "same.e").write_bytes(b"\x00")
    (input_root / "B" / "same.e").write_bytes(b"\x00")

    output_dir = tmp_path / "out_nested"
    exit_code = cli.main(["--in", str(input_root), "--out", str(output_dir), "--glob", "**/*.e"])
    assert exit_code == 0

    assert (output_dir / "A" / "same.edf").exists()
    assert (output_dir / "B" / "same.edf").exists()

    input_root_flat = tmp_path / "inputs_flat"
    input_root_flat.mkdir()
    (input_root_flat / "case.e").write_bytes(b"\x00")
    (input_root_flat / "case.eeg").write_bytes(b"\x00")

    output_dir_flat = tmp_path / "out_flat"
    exit_code = cli.main(["--in", str(input_root_flat), "--out", str(output_dir_flat)])
    assert exit_code == 0

    case_outputs = list(output_dir_flat.glob("case*.edf"))
    assert len(case_outputs) == 2
    assert case_outputs[0].name != case_outputs[1].name


def test_multi_input_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    input_a = tmp_path / "a.e"
    input_b = tmp_path / "b.e"
    input_a.write_bytes(b"\x00")
    input_b.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(input_a), str(input_b), "--out", str(output_dir)])
    assert exit_code == 0
    assert (output_dir / "a.edf").exists()
    assert (output_dir / "b.edf").exists()


def test_ui_single_file_without_real_rich(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    def _fake_run_rich_wizard(*, title: str):
        return type(
            "Args",
            (),
            dict(
                input_paths=[str(recording)],
                output_dir=str(output_dir),
                glob="*.e",
                patient_json=None,
                json_sidecar=False,
                resample_to=None,
                verbose=False,
                ui=True,
            ),
        )()

    def _fake_run_tui(*, inputs, options, convert_one, title: str) -> int:
        for source_path, input_root, output_path in inputs:
            convert_one(source_path=source_path, output_path=output_path, input_root=input_root, status_cb=lambda _: None)
        return 0

    monkeypatch.setattr(cli, "rich_available", lambda: True)
    monkeypatch.setattr(cli, "run_rich_wizard", _fake_run_rich_wizard)
    monkeypatch.setattr(cli, "run_tui", _fake_run_tui)

    exit_code = cli.main(["--ui"])
    assert exit_code == 0
    assert (output_dir / "case.edf").exists()


def test_ui_requires_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "rich_available", lambda: False)
    exit_code = cli.main(["--ui"])
    assert exit_code == 1


def test_cli_rejects_fractional_resample_rate() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--in", "x.e", "--out", "out", "--resample-to", "128.5"])
    assert exc.value.code == 2


def test_convert_rejects_fractional_source_rate_without_resampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.5
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.5,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.5]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.5}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir)])
    assert exit_code == 1
    assert not (output_dir / "case.edf").exists()
