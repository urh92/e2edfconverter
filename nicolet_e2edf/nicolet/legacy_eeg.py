"""Legacy Nicolet/Nervus `.eeg` parsing helpers."""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .types import EventItem, NervusHeader, SegmentInfo, TSEntry

_LEGACY_ANNOTATION_GUID = "{LEGACY-ANNOTATION}"
_BYTES_PER_VALUE = 2

_UINT16 = struct.Struct("<H")
_INT16 = struct.Struct("<h")
_INT32 = struct.Struct("<i")
_UINT32 = struct.Struct("<I")
_FLOAT32 = struct.Struct("<f")


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of file while reading {size} bytes")
    return data


def _read_u16(handle) -> int:
    return _UINT16.unpack(_read_exact(handle, _UINT16.size))[0]


def _read_i16(handle) -> int:
    return _INT16.unpack(_read_exact(handle, _INT16.size))[0]


def _read_i32(handle) -> int:
    return _INT32.unpack(_read_exact(handle, _INT32.size))[0]


def _read_u32(handle) -> int:
    return _UINT32.unpack(_read_exact(handle, _UINT32.size))[0]


def _read_f32(handle) -> float:
    return _FLOAT32.unpack(_read_exact(handle, _FLOAT32.size))[0]


def _read_str(handle, size: int) -> str:
    raw = _read_exact(handle, size)
    return raw.decode("ascii", errors="ignore").split("\x00", 1)[0].strip()


def _normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year if year < 70 else 1900 + year
    if year < 1900:
        # Some legacy files store years as 100-based offsets (e.g., 103 -> 2003).
        return 2000 + (year - 100) if year <= 130 else 1900 + (year - 100)
    return year


def _parse_date(date_parts: list[int]) -> tuple[int, int, int]:
    if len(date_parts) != 3:
        raise ValueError("Legacy date must have three parts")
    a, b, c = date_parts
    if a > 31 and c <= 31:
        year, month, day = a, b, c
    elif c > 31 and a <= 31:
        day, month, year = a, b, c
    else:
        day, month, year = a, b, c
    return _normalize_year(year), max(1, min(12, month)), max(1, min(31, day))


def _build_datetime(date_parts: list[int], time_parts: list[int]) -> datetime:
    try:
        year, month, day = _parse_date(date_parts)
        hour, minute, second = (time_parts + [0, 0, 0])[:3]
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _read_annotations(
    handle, sections_by_id: dict[int, dict], locs_by_id: dict[int, dict]
) -> list[EventItem]:
    section = sections_by_id.get(4)
    loc = locs_by_id.get(4)
    if not section or not loc:
        return []
    handle.seek(loc["start"], 0)

    events: list[EventItem] = []
    for _ in range(int(section.get("rec", 0))):
        ann_type = _read_u16(handle)
        if ann_type == 1:
            _read_u16(handle)
            _read_u16(handle)
            _ = [_read_u16(handle) for _ in range(6)]
            _read_u16(handle)
            _read_u16(handle)
            aux = _read_u32(handle)
            if aux == 1:
                _read_u16(handle)
                _read_u16(handle)
                _read_u16(handle)
                _read_u16(handle)
                handle.seek(12, 1)
            else:
                _read_u16(handle)
                _read_u16(handle)
                _read_u16(handle)
                _read_u16(handle)
                _ = [_read_u16(handle) for _ in range(6)]
            _ = [_read_u16(handle) for _ in range(120)]
            _ = [_read_u16(handle) for _ in range(3)]
        elif ann_type == 2:
            _read_u16(handle)
            _read_u16(handle)
            _ = [_read_u16(handle) for _ in range(6)]
            _ = [_read_u16(handle) for _ in range(6)]
            _read_u16(handle)
            _read_u16(handle)
        elif ann_type == 3:
            _read_u16(handle)
            _read_u16(handle)
            start_time_raw = [_read_u16(handle) for _ in range(3)]
            start_time = list(reversed(start_time_raw))
            start_date = [_read_u16(handle) for _ in range(3)]
            _read_u16(handle)
            _read_u16(handle)
            _read_u16(handle)
            _ = [_read_u16(handle) for _ in range(3)]
            comment = _read_str(handle, 42)
            _ = [_read_u16(handle) for _ in range(64)]
            location = _read_str(handle, 20)
            _ = [_read_u16(handle) for _ in range(36)]
            event_time = _build_datetime(start_date, start_time)
            annotation = comment.strip() if comment else None
            label = location.strip() if location.strip() else "Annotation"
            events.append(
                EventItem(
                    dateOLE=0.0,
                    dateFraction=0.0,
                    date=event_time,
                    duration=0.0,
                    user="",
                    GUID=_LEGACY_ANNOTATION_GUID,
                    label=label,
                    IDStr="Annotation",
                    annotation=annotation,
                    rawLabel=label,
                )
            )
        else:
            break
    return events


def read_legacy_header(path: str | Path) -> NervusHeader:
    filename = Path(path)
    with filename.open("rb") as handle:
        handle.seek(208, 0)
        _ = _read_exact(handle, 30)

        handle.seek(336, 0)
        nr_sec = _read_i16(handle)
        sections = []
        for _ in range(max(nr_sec, 0)):
            name = _read_str(handle, 10)
            _read_u16(handle)
            sec_id = _read_u16(handle)
            length = _read_u16(handle)
            rec = _read_u32(handle)
            _read_u16(handle)
            _read_u16(handle)
            sections.append(
                {"name": name, "id": int(sec_id), "length": int(length), "rec": int(rec)}
            )

        handle.seek(24914, 0)
        _read_i32(handle)
        n_sec = _read_i16(handle)
        locs = []
        for _ in range(max(n_sec, 0)):
            loc_id = _read_i16(handle)
            start = _read_i32(handle)
            length = _read_i32(handle)
            locs.append({"id": int(loc_id), "start": int(start), "length": int(length)})

        if not locs:
            raise ValueError("Legacy file missing section locations")

        locs_by_id = {loc["id"]: loc for loc in locs}
        sections_by_id = {sec["id"]: sec for sec in sections}

        loc0 = locs_by_id.get(0)
        if not loc0:
            raise ValueError("Legacy file missing start timestamp section")
        handle.seek(loc0["start"], 0)
        start_time_raw = [_read_u16(handle) for _ in range(3)]
        start_time = list(reversed(start_time_raw))
        start_date = [_read_u16(handle) for _ in range(3)]
        _ = [_read_u16(handle) for _ in range(3)]
        # End timestamp fields are present in the section layout but currently
        # unused by the converter; read and discard to keep offsets aligned.
        _ = [_read_u16(handle) for _ in range(3)]
        _ = [_read_u16(handle) for _ in range(3)]
        _ = [_read_u16(handle) for _ in range(3)]

        start_dt = _build_datetime(start_date, start_time)

        loc1 = locs_by_id.get(1)
        if not loc1:
            raise ValueError("Legacy file missing channel label section")
        handle.seek(loc1["start"], 0)
        sampling_rate = float(_read_i16(handle))
        nr_traces = int(_read_i16(handle))
        nr_channels = int(_read_i16(handle))
        if sampling_rate <= 0 or nr_traces <= 0:
            raise ValueError("Legacy file reports invalid sampling rate or trace count")

        chan_info = []
        for _ in range(nr_traces):
            name = _read_str(handle, 6)
            _read_u16(handle)
            reference = _read_str(handle, 6)
            mult = _read_f32(handle)
            _ = [_read_u16(handle) for _ in range(7)]
            conv = _read_f32(handle)
            _ = [_read_f32(handle) for _ in range(3)]
            chan_info.append(
                {
                    "name": name.strip(),
                    "reference": reference.strip(),
                    "mult": float(mult),
                    "conv": float(conv),
                }
            )

        montage_info = []
        loc3 = locs_by_id.get(3)
        if loc3:
            handle.seek(loc3["start"], 0)
            montage_name = _read_str(handle, 32)
            _read_u16(handle)
            nr_ch = _read_u16(handle)
            for _ in range(int(nr_ch)):
                deriv_name = _read_str(handle, 12)
                _ = [_read_f32(handle) for _ in range(5)]
                color = _read_exact(handle, 3)
                handle.seek(1, 1)
                _ = [_read_u16(handle) for _ in range(3)]
                _conv = _read_f32(handle)
                montage_info.append(
                    {
                        "montageName": montage_name.strip(),
                        "derivationName": deriv_name.strip(),
                        "signalName1": deriv_name.strip(),
                        "signalName2": "",
                        "color": int.from_bytes(color, "little"),
                    }
                )

        events = _read_annotations(handle, sections_by_id, locs_by_id)

        data_loc = max(locs, key=lambda loc: loc["id"])
        data_start = int(data_loc["start"])
        data_length = int(data_loc["length"])

    nr_values = data_length // (_BYTES_PER_VALUE * nr_traces)
    if nr_values <= 0:
        raise ValueError("Legacy file contains no waveform data")

    labels = [ch.get("name") or f"Ch{idx+1}" for idx, ch in enumerate(chan_info)]
    refs = [ch.get("reference") or "" for ch in chan_info]
    conv = np.array([ch.get("conv", 1.0) for ch in chan_info], dtype=float)
    conv[np.isclose(conv, 0.0)] = 1.0
    sampling = np.full(nr_traces, sampling_rate, dtype=float)
    sample_counts = np.full(nr_traces, int(nr_values), dtype=int)
    eeg_offset = np.zeros(nr_traces, dtype=float)

    ts_entries = [
        TSEntry(
            label=labels[idx],
            activeSensor=labels[idx],
            refSensor=refs[idx],
            lowcut=0.0,
            hiCut=0.0,
            samplingRate=sampling_rate,
            resolution=float(conv[idx]),
            specialMark="",
            notch=False,
            eeg_offset=0,
        )
        for idx in range(nr_traces)
    ]

    segment = SegmentInfo(
        dateOLE=0.0,
        date=start_dt,
        duration=float(nr_values) / sampling_rate,
        chName=labels,
        refName=refs,
        samplingRate=sampling,
        scale=conv,
        sampleCount=sample_counts,
        eegOffset=eeg_offset,
    )

    channel_info = [
        {
            "sensor": labels[idx],
            "samplingRate": sampling_rate,
            "bOn": True,
            "lInputID": 0,
            "lInputSettingID": 0,
            "indexID": idx,
        }
        for idx in range(nr_traces)
    ]

    nrv_header = NervusHeader(
        filename=filename,
        PatientInfo=None,
        SigInfo=[],
        ChannelInfo=channel_info,
        TSInfo=ts_entries,
        TSInfoBySegment=[ts_entries],
        Segments=[segment],
        Events=events,
        MontageInfo=montage_info,
        LegacyInfo={
            "sections": sections,
            "locs": locs,
            "samplingRate": sampling_rate,
            "nrTraces": nr_traces,
            "nrChannels": nr_channels,
            "nrValues": nr_values,
            "dataStart": data_start,
            "conv": conv.tolist(),
        },
        format="nervus-eeg",
        startDateTime=start_dt,
    )
    return nrv_header


def read_legacy_data(
    path: str | Path,
    header: NervusHeader,
    channels: list[int] | None = None,
    begsample: int | None = None,
    endsample: int | None = None,
) -> np.ndarray:
    if not header.LegacyInfo:
        raise ValueError("Legacy header missing waveform metadata")

    nr_traces = int(header.LegacyInfo.get("nrTraces", 0) or 0)
    nr_values = int(header.LegacyInfo.get("nrValues", 0) or 0)
    data_start = int(header.LegacyInfo.get("dataStart", 0) or 0)
    conv = header.LegacyInfo.get("conv") or [1.0] * nr_traces

    if nr_traces <= 0 or nr_values <= 0:
        raise ValueError("Legacy header does not describe valid data")

    if channels is None:
        if header.matchingChannels:
            channel_selection = sorted(header.matchingChannels)
        else:
            channel_selection = list(range(1, nr_traces + 1))
    else:
        channel_selection = sorted(set(int(ch) for ch in channels))
        if any(ch < 1 for ch in channel_selection):
            raise ValueError("Channel indices must be 1-based and positive")

    beg = 1 if begsample is None else int(begsample)
    end = nr_values if endsample is None else int(endsample)
    if beg < 1 or end < beg:
        raise ValueError("Invalid sample range specified")

    beg_zero = beg - 1
    end_exclusive = min(end, nr_values)
    requested = end_exclusive - beg_zero
    if requested <= 0:
        return np.empty((len(channel_selection), 0), dtype=np.float32)

    with Path(path).open("rb") as handle:
        handle.seek(data_start + beg_zero * nr_traces * _BYTES_PER_VALUE, 0)
        raw = np.fromfile(handle, dtype="<i2", count=requested * nr_traces)

    usable = min(requested, raw.size // nr_traces)
    if usable <= 0:
        return np.empty((len(channel_selection), 0), dtype=np.float32)

    raw = raw[: usable * nr_traces].reshape(usable, nr_traces)
    channel_indices = [ch - 1 for ch in channel_selection if ch - 1 < nr_traces]
    data = np.zeros((requested, len(channel_indices)), dtype=np.float32)
    if channel_indices:
        selected = raw[:, channel_indices].astype(np.float32, copy=False)
        for idx, ch in enumerate(channel_indices):
            factor = float(conv[ch]) if ch < len(conv) else 1.0
            if np.isclose(factor, 0.0):
                factor = 1.0
            data[:usable, idx] = selected[:, idx] * factor
    return data.T
