from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from .header import canonical_event_text
from .types import EventItem


# EDF+ specification month abbreviations (must be uppercase, exactly 3 chars)
_MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_SFREQ_INT_TOL = 1e-6


def _require_integer_sampling_rate(sfreq: float) -> int:
    """Validate that EDF output sampling rate is an integer Hz value."""
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be a positive finite number")
    rounded = int(round(float(sfreq)))
    if rounded <= 0 or not np.isclose(float(sfreq), float(rounded), rtol=0.0, atol=_SFREQ_INT_TOL):
        raise ValueError(
            f"EDF output requires an integer sampling rate in Hz, got {sfreq}. "
            "Use resampling to an integer Hz rate before writing."
        )
    return rounded


def _sanitize_subfield(text: str) -> str:
    """Sanitize a subfield for EDF+ structured fields.
    
    EDF+ uses spaces as delimiters in patient/recording fields.
    Subfields containing spaces must use underscores instead.
    Only printable ASCII 32-126 allowed, but spaces become underscores.
    """
    # Replace spaces with underscores per EDF+ spec
    text = text.replace(" ", "_")
    # Keep only printable ASCII (32-126), but we already replaced spaces
    cleaned = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    return cleaned or "X"


def _format_edfplus_patient(patient_meta: dict[str, str]) -> str:
    """Format patient identification per EDF+ spec.
    
    EDF+ requires: <patient_code> <sex> <birthdate> <patient_name>
    - Spaces are used as delimiters between subfields
    - Subfields use underscores for internal spaces
    - Unknown fields should use 'X'
    - Sex: M, F, or X (unknown)
    - Birthdate: DD-MMM-YYYY (e.g., 02-MAY-1951) or X
    """
    # Patient code (ID)
    patient_code = _sanitize_subfield(patient_meta.get("PatientID", "X"))
    
    # Sex: M, F, or X
    sex_raw = patient_meta.get("PatientSex", "X").upper().strip()
    if sex_raw in ("M", "MALE"):
        sex = "M"
    elif sex_raw in ("F", "FEMALE"):
        sex = "F"
    else:
        sex = "X"
    
    # Birthdate: DD-MMM-YYYY or X
    birthdate = "X"
    birthdate_raw = patient_meta.get("PatientBirthDate", "")
    if birthdate_raw:
        # Try to parse common date formats
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y%m%d"):
            try:
                dt = datetime.strptime(birthdate_raw, fmt)
                birthdate = f"{dt.day:02d}-{_MONTH_ABBR[dt.month - 1]}-{dt.year}"
                break
            except ValueError:
                continue
    
    # Patient name (use underscores for spaces)
    patient_name = _sanitize_subfield(patient_meta.get("PatientName", "X"))
    
    # Combine with single spaces as delimiters
    return f"{patient_code} {sex} {birthdate} {patient_name}"


def _format_edfplus_recording(recording_start: datetime, patient_meta: dict[str, str]) -> str:
    """Format recording identification per EDF+ spec.
    
    EDF+ requires: Startdate <DD-MMM-YYYY> <admin_code> <technician> <equipment>
    - Must start with literal "Startdate " followed by the date
    - Date format: DD-MMM-YYYY (e.g., 02-MAR-2002)
    - Unknown subfields use 'X'
    """
    # Format date as DD-MMM-YYYY
    date_str = f"{recording_start.day:02d}-{_MONTH_ABBR[recording_start.month - 1]}-{recording_start.year}"
    
    # Admin code (use study description or X)
    admin_code = _sanitize_subfield(
        patient_meta.get("StudyDescription") or 
        patient_meta.get("SeriesDescription") or 
        "X"
    )
    
    # Technician (unknown)
    technician = "X"
    
    # Equipment identifier
    equipment = "nicolet-e2edf"
    
    return f"Startdate {date_str} {admin_code} {technician} {equipment}"


def _pad(text: str, length: int) -> bytes:
    data = text.encode("ascii", "ignore")[:length]
    return data.ljust(length, b" ")


def _format_float(value: float, length: int, decimals: int = 3) -> bytes:
    """Format a float for EDF header fields.
    
    EDF spec requires ASCII representation within fixed width.
    We prefer simpler integer representation when the value is a whole number.
    """
    # If the value is effectively an integer, write it as such
    if value == int(value):
        text = str(int(value))
        if len(text) <= length:
            return _pad(text, length)
    
    # Otherwise, try with decreasing precision
    text = f"{value:.{decimals}f}"
    if len(text) > length:
        text = f"{value:.1f}"
    if len(text) > length:
        text = f"{int(value)}"
    return _pad(text, length)


def _channel_label(name: str, index: int) -> str:
    cleaned = name.split("\x00", 1)[0].strip()
    cleaned = "".join(ch for ch in cleaned if 32 <= ord(ch) <= 126)
    if not cleaned:
        cleaned = f"Ch{index}"
    return cleaned[:16]


def _format_time_keeper_tal(record_start_time: float) -> bytes:
    """Format the time-keeping TAL for a data record.
    
    EDF+ requires each data record to start with a time-keeping TAL.
    Format: +<onset>\x14\x14\x00 (onset with empty annotation)
    """
    # Format onset with sign (+ or -) as required by EDF+
    # Use minimal precision to save space
    if record_start_time == int(record_start_time):
        onset = f"+{int(record_start_time)}"
    else:
        onset = f"+{record_start_time:.6f}".rstrip('0').rstrip('.')
    return f"{onset}\x14\x14\x00".encode("ascii")


def _format_event_tal(event: EventItem, recording_start: datetime) -> bytes:
    """Format a single event as a TAL."""
    onset_seconds = (event.date - recording_start).total_seconds()
    # Format onset with sign (+ or -) as required by EDF+
    onset = f"{onset_seconds:+.6f}"
    
    # Duration is optional, format: \x15<duration>
    duration = ""
    if event.duration is not None and event.duration > 0:
        duration = f"\x15{event.duration:.6f}"
    
    event_id, label, annotation = canonical_event_text(event)
    description_parts = []
    if label:
        description_parts.append(label)
    if annotation:
        description_parts.append(annotation)
    elif event_id and not description_parts:
        description_parts.append(event_id)
    text = ": ".join(part for part in description_parts if part) or "Event"
    
    # TAL format: <onset><duration>\x14<annotation>\x14\x00
    return f"{onset}{duration}\x14{text}\x14\x00".encode("utf-8", "ignore")


def _build_annotation_signal(tal_bytes: bytes, annotation_samples: int) -> np.ndarray:
    """Build the annotation signal array from TAL bytes.
    
    The TAL bytes are stored directly as raw bytes within int16 samples.
    Each int16 sample holds 2 bytes of TAL data (or zero-padding).
    """
    # Pad to even length for int16 alignment
    if len(tal_bytes) % 2 != 0:
        tal_bytes = tal_bytes + b'\x00'
    
    # Create signal array with specified size (may be larger than needed for padding)
    signal = np.zeros(annotation_samples, dtype=np.int16)
    
    # Copy TAL bytes into signal
    n_samples_needed = len(tal_bytes) // 2
    if n_samples_needed > 0:
        signal[:n_samples_needed] = np.frombuffer(tal_bytes, dtype='<i2')
    
    return signal


def _distribute_events_to_records(
    events: Sequence[EventItem] | None,
    recording_start: datetime,
    n_records: int,
    record_duration: float,
) -> list[list[EventItem]]:
    """Distribute events to their appropriate data records based on onset time.
    
    Returns a list of event lists, one per data record.
    """
    events_per_record: list[list[EventItem]] = [[] for _ in range(n_records)]
    
    if not events:
        return events_per_record
    
    for event in events:
        if event.date is None:
            continue
        onset_seconds = (event.date - recording_start).total_seconds()
        # Determine which record this event belongs to
        record_idx = int(onset_seconds / record_duration)
        # Clamp to valid range (events at or after end go to last record)
        record_idx = max(0, min(record_idx, n_records - 1))
        events_per_record[record_idx].append(event)
    
    return events_per_record


def _calculate_max_annotation_samples(
    events: Sequence[EventItem] | None,
    recording_start: datetime,
    n_records: int,
    record_duration: float,
) -> int:
    """Calculate the maximum number of annotation samples needed per record.
    
    All records must have the same number of annotation samples (EDF requirement).
    We calculate the worst-case size and use that for all records.
    """
    events_per_record = _distribute_events_to_records(
        events, recording_start, n_records, record_duration
    )
    
    max_bytes = 0
    for record_idx, record_events in enumerate(events_per_record):
        # Time-keeper TAL for this record
        record_start_time = record_idx * record_duration
        time_keeper = _format_time_keeper_tal(record_start_time)
        
        # Event TALs for this record
        event_tals = b"".join(
            _format_event_tal(event, recording_start)
            for event in record_events
        )
        
        total_bytes = len(time_keeper) + len(event_tals)
        max_bytes = max(max_bytes, total_bytes)
    
    # Minimum size for time-keeper TAL even with no events
    min_bytes = len(_format_time_keeper_tal(0.0))
    max_bytes = max(max_bytes, min_bytes)
    
    # Convert bytes to samples (2 bytes per sample, round up)
    return (max_bytes + 1) // 2


def write_edf(
    output_path: str | Path,
    data_uV: np.ndarray | Sequence[np.ndarray],
    sfreq: float | Sequence[float],
    ch_names: Sequence[str],
    patient_meta: dict[str, str] | None = None,
    recording_start: datetime | None = None,
    annotations: Sequence[EventItem] | None = None,
) -> Path:
    """Write an EDF+ compatible file from microvolt data.

    Supports both uniform and mixed sampling rates:
    - Uniform: pass a 2D array [samples, channels] and a single float sfreq.
    - Mixed:   pass a list of 1D arrays (one per channel) and a list of floats
               (one sampling rate per channel). Each signal is stored at its
               native rate; no upsampling is performed.

    Outputs files compliant with the EDF+ specification:
    https://www.edfplus.info/specs/edfplus.html

    Key EDF+ compliance features:
    - Uses 1-second data records to stay well under size limits
    - Reserved field starts with 'EDF+C' (continuous recording)
    - Patient identification uses structured format: code sex birthdate name
    - Recording identification starts with 'Startdate DD-MMM-YYYY'
    - EDF Annotations signal for events with time-keeping TAL per record
    """
    # --- Normalise inputs to per-channel representation ---
    if isinstance(sfreq, (int, float)):
        # Scalar sfreq: data_uV must be 2D [samples, channels]
        if not isinstance(data_uV, np.ndarray) or data_uV.ndim != 2:
            raise ValueError("EDF writer expects data shaped as [samples, channels]")
        n_samples, n_channels = data_uV.shape
        ch_arrays: list[np.ndarray] = [data_uV[:, i] for i in range(n_channels)]
        ch_sfreqs: list[float] = [float(sfreq)] * n_channels
    else:
        # Per-channel sfreqs
        ch_sfreqs = [float(f) for f in sfreq]
        n_channels = len(ch_sfreqs)
        if isinstance(data_uV, np.ndarray) and data_uV.ndim == 2:
            ch_arrays = [data_uV[:, i] for i in range(n_channels)]
        else:
            ch_arrays = [np.asarray(d, dtype=np.float32).ravel() for d in data_uV]

    if n_channels != len(ch_names):
        raise ValueError("Channel-name list must match waveform column count")
    if n_channels == 0:
        raise ValueError("No channels to write")
    if any(f <= 0 for f in ch_sfreqs):
        raise ValueError("Sampling frequency must be positive for all channels")

    # Guard against NaN/Inf from upstream scaling to keep EDF ranges finite.
    for i in range(n_channels):
        if not np.isfinite(ch_arrays[i]).all():
            ch_arrays[i] = np.where(np.isfinite(ch_arrays[i]), ch_arrays[i], 0.0)

    patient_meta = patient_meta or {}
    start = recording_start or datetime.utcnow()

    # EDF+ compliant patient identification field
    # Format: <code> <sex> <birthdate> <name> with underscores for spaces in subfields
    patient_field = _format_edfplus_patient(patient_meta)

    # EDF+ compliant recording identification field
    # Format: Startdate DD-MMM-YYYY <admin_code> <technician> <equipment>
    recording_field = _format_edfplus_recording(start, patient_meta)

    # Use 1-second data records to stay well under EDF size limits.
    # Each channel declares its own samples-per-record (= its Hz rounded to int),
    # so mixed-rate signals are stored natively without upsampling.
    record_duration = 1.0  # seconds per record
    samples_per_record: list[int] = [max(1, int(round(f))) for f in ch_sfreqs]

    # Total recording duration driven by the longest channel.
    total_duration = max(
        (len(ch_arrays[i]) / ch_sfreqs[i] if ch_sfreqs[i] > 0 else 0.0)
        for i in range(n_channels)
    )
    if total_duration <= 0:
        raise ValueError("Sampling frequency and data must be non-zero")
    n_records = int(np.ceil(total_duration / record_duration))
    
    # Include annotations signal if annotations are provided
    include_annotations = annotations is not None
    
    # Calculate annotation signal size (must be same for all records)
    annotation_samples = 0
    if include_annotations:
        annotation_samples = _calculate_max_annotation_samples(
            annotations, start, n_records, record_duration
        )
        # Ensure minimum size for time-keeper TAL
        annotation_samples = max(annotation_samples, 8)

    total_channels = n_channels + (1 if include_annotations else 0)
    header_bytes = 256 + total_channels * 256
    
    # Date format: DD.MM.YY per EDF spec
    # Note: 2-digit year uses 1985 clipping (85-99 = 1985-1999, 00-84 = 2000-2084)
    start_date = start.strftime("%d.%m.%y")
    start_time = start.strftime("%H.%M.%S")

    # Build the 256-byte fixed header
    header = bytearray()
    
    # Version: 8 bytes, must be "0" followed by spaces
    header.extend(_pad("0", 8))
    
    # Patient identification: 80 bytes, EDF+ structured format
    header.extend(_pad(patient_field, 80))
    
    # Recording identification: 80 bytes, EDF+ structured format
    header.extend(_pad(recording_field, 80))
    
    # Start date: 8 bytes, DD.MM.YY
    header.extend(_pad(start_date, 8))
    
    # Start time: 8 bytes, HH.MM.SS
    header.extend(_pad(start_time, 8))
    
    # Number of bytes in header: 8 bytes
    header.extend(_pad(str(header_bytes), 8))
    
    # Reserved: 44 bytes
    # EDF+ REQUIRES this field to start with "EDF+C" (continuous) or "EDF+D" (discontinuous)
    # Only mark as EDF+ if we're including the EDF Annotations signal
    # Without annotations, we write plain EDF (empty reserved field)
    if include_annotations:
        header.extend(_pad("EDF+C", 44))
    else:
        header.extend(_pad("", 44))
    
    # Number of data records: 8 bytes
    header.extend(_pad(str(n_records), 8))
    
    # Duration of data record in seconds: 8 bytes
    header.extend(_format_float(record_duration, 8, decimals=5))
    
    # Number of signals: 4 bytes
    header.extend(_pad(str(total_channels), 4))

    # Build signal labels
    labels = [_channel_label(name, idx + 1) for idx, name in enumerate(ch_names)]
    
    # Calculate physical ranges from actual data (per channel)
    physical_min = np.array(
        [np.min(d) if d.size > 0 else 0.0 for d in ch_arrays]
    )
    physical_max = np.array(
        [np.max(d) if d.size > 0 else 1.0 for d in ch_arrays]
    )

    # EDF spec requires physical_min < physical_max (strictly less than).
    # For constant signals (e.g., all zeros), add a tiny offset to max.
    equal_mask = physical_min == physical_max
    if np.any(equal_mask):
        physical_max = np.where(equal_mask, physical_max + 1.0, physical_max)

    physical_diff = np.maximum(physical_max - physical_min, 1.0)

    # Digital range: full 16-bit signed range
    digital_min = np.full(n_channels, -32768, dtype=np.int32)
    digital_max = np.full(n_channels, 32767, dtype=np.int32)
    samples_per_record_array = np.array(samples_per_record, dtype=np.int32)

    if include_annotations:
        # EDF Annotations signal per EDF+ spec section 2.2.1
        labels.append("EDF Annotations")
        
        # EDF+ spec: annotation signal uses physical range -1 to 1
        # and digital range -32768 to 32767
        # This allows TAL bytes (0-127 ASCII) to be stored correctly
        physical_min = np.concatenate([physical_min, np.array([-1.0])])
        physical_max = np.concatenate([physical_max, np.array([1.0])])
        physical_diff = np.concatenate([physical_diff, np.array([2.0])])  # 1 - (-1) = 2
        digital_min = np.concatenate([digital_min, np.array([-32768], dtype=np.int32)])
        digital_max = np.concatenate([digital_max, np.array([32767], dtype=np.int32)])
        samples_per_record_array = np.concatenate([
            samples_per_record_array, 
            np.array([annotation_samples], dtype=np.int32)
        ])

    # Build signal header sections per EDF spec
    # Each section contains one field for each signal, stored sequentially
    sections = [
        # Signal labels (16 bytes each)
        (labels, lambda text: _pad(text, 16)),
        # Transducer type (80 bytes each) - empty for now
        (["" for _ in range(total_channels)], lambda text: _pad(text, 80)),
        # Physical dimension (8 bytes each)
        (["uV" for _ in range(n_channels)] + ["" for _ in range(total_channels - n_channels)], lambda text: _pad(text, 8)),
        # Physical minimum (8 bytes each)
        (physical_min, lambda value: _format_float(float(value), 8)),
        # Physical maximum (8 bytes each)
        (physical_max, lambda value: _format_float(float(value), 8)),
        # Digital minimum (8 bytes each)
        (digital_min, lambda value: _pad(str(int(value)), 8)),
        # Digital maximum (8 bytes each)
        (digital_max, lambda value: _pad(str(int(value)), 8)),
        # Prefiltering info (80 bytes each) - empty per EDF spec (can be filled if known)
        (["" for _ in range(total_channels)], lambda text: _pad(text, 80)),
        # Number of samples per data record (8 bytes each)
        (samples_per_record_array, lambda value: _pad(str(int(value)), 8)),
        # Reserved (32 bytes each) - must be empty
        (["" for _ in range(total_channels)], lambda text: _pad(text, 32)),
    ]

    for values, formatter in sections:
        for value in values:
            header.extend(formatter(value))

    # Pre-compute scaling factors for digital conversion
    digital_range = (digital_max - digital_min).astype(np.float64)
    scales = digital_range / physical_diff.astype(np.float64)
    
    # Pre-distribute annotations to records if needed
    events_per_record = None
    if include_annotations:
        events_per_record = _distribute_events_to_records(
            annotations, start, n_records, record_duration
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("wb") as handle:
        # Write header
        handle.write(header)
        
        # Write data records
        # EDF format: for each record, write all channels sequentially.
        # Each channel uses its own samples-per-record (native rate support).
        for record_idx in range(n_records):
            # Write each signal's data for this record
            for ch_idx in range(n_channels):
                spr = samples_per_record[ch_idx]
                ch_start = record_idx * spr
                ch_end = min(ch_start + spr, len(ch_arrays[ch_idx]))
                channel_data = ch_arrays[ch_idx][ch_start:ch_end]
                actual_samples = len(channel_data)

                # Scale to digital values
                scaled = (
                    (channel_data - physical_min[ch_idx]) * scales[ch_idx] + digital_min[ch_idx]
                )
                signal = np.clip(np.rint(scaled), -32768, 32767).astype("<i2")

                # Pad with zeros if this is a partial record (last record)
                if actual_samples < spr:
                    padded = np.zeros(spr, dtype="<i2")
                    padded[:actual_samples] = signal
                    signal = padded

                handle.write(signal.tobytes())
            
            # Write annotation signal for this record if included
            if include_annotations and events_per_record is not None:
                # Build TAL bytes for this record
                record_start_time = record_idx * record_duration
                time_keeper = _format_time_keeper_tal(record_start_time)
                
                # Add event TALs for events in this record
                event_tals = b"".join(
                    _format_event_tal(event, start)
                    for event in events_per_record[record_idx]
                )
                
                tal_bytes = time_keeper + event_tals
                annotation_signal = _build_annotation_signal(tal_bytes, annotation_samples)
                handle.write(annotation_signal.tobytes())
    
    return output_path
