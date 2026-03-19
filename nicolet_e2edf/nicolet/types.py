# This file includes logic adapted from FieldTrip's read_nervus_header.m and
# read_nervus_data.m. FieldTrip is released under the GPL-3.0 licence.
# Copyright (C) the FieldTrip project.

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}

@dataclass(**DATACLASS_KWARGS)
class StaticPacket:
    """Representation of a static packet entry within the `.e` file."""

    tag: str
    index: int
    IDStr: str


@dataclass(**DATACLASS_KWARGS)
class MainIndexEntry:
    """Entry in the main index pointing to sections of the `.e` container."""

    sectionIdx: int
    offset: int
    blockL: int
    sectionL: int


@dataclass(**DATACLASS_KWARGS)
class TSEntry:
    """Time-series (TS) entry describing a logical EEG signal."""

    label: str
    activeSensor: str
    refSensor: str
    lowcut: float
    hiCut: float
    samplingRate: float
    resolution: float
    specialMark: str
    notch: bool
    eeg_offset: float


@dataclass(**DATACLASS_KWARGS)
class SegmentInfo:
    """Segment data describing a continuous portion of the recording."""

    dateOLE: float
    date: datetime
    duration: float
    chName: list[str]
    refName: list[str]
    samplingRate: np.ndarray
    scale: np.ndarray
    sampleCount: np.ndarray
    eegOffset: np.ndarray | None = None


@dataclass(**DATACLASS_KWARGS)
class EventItem:
    """Event extracted from the `.e` file with optional annotation text."""

    dateOLE: float
    dateFraction: float
    date: datetime
    duration: float
    user: str
    GUID: str
    label: str
    IDStr: str
    annotation: str | None = None
    segmentIndex: int | None = None
    isEpoch: bool | None = None
    rawLabel: str | None = None


@dataclass(**DATACLASS_KWARGS)
class SleepEpoch:
    """A single scored sleep epoch."""

    stage: int
    duration: int
    index: int


@dataclass(**DATACLASS_KWARGS)
class SleepScoreInfo:
    """A complete hypnogram from one scoring session."""

    algorithm: str
    epoch_duration: int
    epochs: list[SleepEpoch]


@dataclass(**DATACLASS_KWARGS)
class NervusHeader:
    """Composite header structure compatible with FieldTrip expectations."""

    filename: Path
    StaticPackets: list[StaticPacket] = field(default_factory=list)
    QIIndex: dict[str, Any] | None = None
    QIIndex2: list[dict[str, Any]] = field(default_factory=list)
    MainIndex: list[MainIndexEntry] = field(default_factory=list)
    allIndexIDs: list[int] = field(default_factory=list)
    infoGuids: list[dict[str, Any]] = field(default_factory=list)
    DynamicPackets: list[dict[str, Any]] = field(default_factory=list)
    PatientInfo: dict[str, Any] | None = None
    SigInfo: list[dict[str, Any]] = field(default_factory=list)
    ChannelInfo: list[dict[str, Any]] = field(default_factory=list)
    # Parsed from INPUTGUID / INPUTSETTINGSGUID when available. These are kept
    # as generic decoded tables because field semantics vary across recordings.
    InputInfo: list[dict[str, Any]] = field(default_factory=list)
    InputSettingsInfo: list[dict[str, Any]] = field(default_factory=list)
    TSInfo: list[TSEntry] = field(default_factory=list)
    TSInfoBySegment: list[list[TSEntry]] = field(default_factory=list)
    Segments: list[SegmentInfo] = field(default_factory=list)
    Events: list[EventItem] = field(default_factory=list)
    EventTypeInfo: dict[str, str] = field(default_factory=dict)
    MontageInfo: list[dict[str, Any]] = field(default_factory=list)
    MontageInfo2: list[dict[str, Any]] = field(default_factory=list)
    LegacyInfo: dict[str, Any] = field(default_factory=dict)
    SleepScores: list[SleepScoreInfo] = field(default_factory=list)
    format: str | None = None  # "nicolet-e", "nervus-eeg", etc.
    reference: str | None = None
    targetSamplingRate: float | None = None
    matchingChannels: list[int] = field(default_factory=list)
    excludedChannels: list[int] = field(default_factory=list)
    targetNumberOfChannels: int | None = None
    targetSampleCount: int | None = None
    startDateTime: datetime | None = None
