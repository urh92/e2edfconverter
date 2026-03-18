# This file includes logic adapted from FieldTrip's read_nervus_data.m.
# FieldTrip is released under the GPL-3.0 licence. Copyright (C) the FieldTrip project.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .types import NervusHeader


def _normalise_channel_selection(header: NervusHeader, channels: Iterable[int] | None) -> list[int]:
    """Return a 1-based sorted list of channels to read."""

    if channels is None:
        if header.matchingChannels:
            return sorted(header.matchingChannels)
        return list(range(1, len(header.TSInfo) + 1))
    channel_list = sorted(set(int(ch) for ch in channels))
    if any(ch < 1 for ch in channel_list):
        raise ValueError("Channel indices must be 1-based and positive")
    return channel_list


def _lookup_static_index(header: NervusHeader, channel_zero_based: int) -> int:
    target_tag = str(channel_zero_based)
    for packet in header.StaticPackets:
        if packet.tag.strip() == target_tag:
            return packet.index
    raise KeyError(f"Static packet for channel {channel_zero_based + 1} not found")


def _collect_sections(header: NervusHeader, section_idx: int):
    sections = [entry for entry in header.MainIndex if entry.sectionIdx == section_idx]
    lengths = [entry.sectionL // 2 for entry in sections]
    cumulative = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return sections, np.array(lengths, dtype=np.int64), cumulative


def _accumulate_segment_lengths(header: NervusHeader, channel_zero_based: int) -> np.ndarray:
    lengths = [int(segment.sampleCount[channel_zero_based]) for segment in header.Segments]
    return np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))


def _read_section_chunk(handle, entry, start_offset_samples: int, count: int) -> np.ndarray:
    handle.seek(entry.offset + start_offset_samples * 2, 0)
    return np.fromfile(handle, dtype="<i2", count=count)


def _read_channel_window(
    handle,
    sections,
    cumulative_lengths,
    start_sample: int,
    count: int,
) -> np.ndarray:
    """Read a window of samples for a single channel, handling multi-section storage."""

    if count <= 0:
        return np.empty(0, dtype=np.int16)
    if not sections:
        return np.empty(0, dtype=np.int16)
    output = np.empty(count, dtype=np.int16)
    written = 0
    target_start = start_sample
    target_end = start_sample + count
    start_idx = int(np.searchsorted(cumulative_lengths, target_start, side="right") - 1)
    if start_idx < 0:
        start_idx = 0
    if start_idx >= len(sections):
        start_idx = len(sections) - 1
    for idx in range(start_idx, len(sections)):
        entry = sections[idx]
        section_start = cumulative_lengths[idx]
        section_end = cumulative_lengths[idx + 1]
        if target_end <= section_start:
            break
        if target_start >= section_end:
            continue
        read_start = max(target_start, section_start)
        read_end = min(target_end, section_end)
        samples_to_read = read_end - read_start
        if samples_to_read <= 0:
            continue
        offset_within_section = read_start - section_start
        chunk = _read_section_chunk(handle, entry, offset_within_section, samples_to_read)
        actual = chunk.size
        if actual == 0:
            break
        output[written : written + actual] = chunk
        written += actual
        if actual < samples_to_read:
            break
    return output[:written]


def read_nervus_data(
    path: str | Path,
    header: NervusHeader,
    channels: Iterable[int] | None = None,
    begsample: int | None = None,
    endsample: int | None = None,
) -> np.ndarray:
    """Read waveform data from a Nicolet/Nervus recording."""

    if header.format == "nervus-eeg":
        from .legacy_eeg import read_legacy_data

        return read_legacy_data(path, header, channels=channels, begsample=begsample, endsample=endsample)
    if header.format == "nervus-legacy-e":
        from .legacy import read_legacy_data

        return read_legacy_data(path, header, channels=channels, begsample=begsample, endsample=endsample)

    if not header.Segments:
        raise ValueError("Header does not contain segment information")
    channel_selection = _normalise_channel_selection(header, channels)
    zero_based_channels = [ch - 1 for ch in channel_selection]
    base_channel = zero_based_channels[0]
    cumulative_segment_lengths = _accumulate_segment_lengths(header, base_channel)
    total_samples = int(cumulative_segment_lengths[-1])
    beg = 1 if begsample is None else int(begsample)
    end = total_samples if endsample is None else int(endsample)
    if beg < 1 or end < beg:
        raise ValueError("Invalid sample range specified")
    beg_zero = beg - 1
    end_exclusive = min(end, total_samples)
    samples_requested = end_exclusive - beg_zero
    data = np.zeros((len(zero_based_channels), samples_requested), dtype=np.float32)
    sections_cache: dict[int, tuple] = {}
    offsets_cache: dict[int, np.ndarray] = {}
    with Path(path).open("rb") as handle:
        # Iterate per channel so we can reuse section metadata and honour scaling.
        for ch_idx, channel_zb in enumerate(zero_based_channels):
            offsets = offsets_cache.get(channel_zb)
            if offsets is None:
                offsets = _accumulate_segment_lengths(header, channel_zb)
                offsets_cache[channel_zb] = offsets
            if not np.array_equal(offsets, cumulative_segment_lengths):
                raise NotImplementedError(
                    "Mixed sampling rates across requested channels are not yet supported."
                )
            # Collect all MainIndex sections that belong to this channel's data stream.
            sections = sections_cache.get(channel_zb)
            if sections is None:
                section_idx = _lookup_static_index(header, channel_zb)
                sections_cache[channel_zb] = _collect_sections(header, section_idx)
            entries, _section_lengths, cumulative_lengths = sections_cache[channel_zb]
            for seg_idx, segment in enumerate(header.Segments):
                # Translate the global sample window into the portion stored inside this segment.
                segment_start = cumulative_segment_lengths[seg_idx]
                segment_end = cumulative_segment_lengths[seg_idx + 1]
                overlap_start = max(beg_zero, segment_start)
                overlap_end = min(end_exclusive, segment_end)
                if overlap_start >= overlap_end:
                    continue
                relative_start = overlap_start - beg_zero
                samples_to_copy = overlap_end - overlap_start
                channel_offset = offsets[seg_idx]
                window_start = channel_offset + (overlap_start - segment_start)
                raw = _read_channel_window(
                    handle,
                    entries,
                    cumulative_lengths,
                    int(window_start),
                    int(samples_to_copy),
                )
                if raw.size == 0:
                    break
                scale = float(segment.scale[channel_zb]) if segment.scale is not None else 1.0
                if np.isclose(scale, 0.0):
                    scale = 1.0
                offset = 0.0
                if segment.eegOffset is not None and channel_zb < len(segment.eegOffset):
                    offset = float(segment.eegOffset[channel_zb])
                    if not np.isfinite(offset) or abs(offset) > 1e9:
                        offset = 0.0
                count = min(raw.size, samples_to_copy)
                data[ch_idx, relative_start : relative_start + count] = raw[:count] * scale + offset
                if raw.size < samples_to_copy:
                    break
    return data
