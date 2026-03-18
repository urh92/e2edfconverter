# Nicolet `.e`/`.eeg` → EDF

<img src="docs/logo.png" alt="Logo" width="200">

A Python tool to convert Nicolet/Nervus `.e` EEG files into standard EDF+ format. No vendor DLLs, no MATLAB (which costs money!), just Python! I couldn't find a native Python way to get `.e` files out of their vendor format, so me and Opus 4.5 wrote this.

> **Acknowledgment**: This project wouldn't exist without the excellent [FieldTrip](https://github.com/fieldtrip/fieldtrip) toolbox. Their MATLAB implementation of the Nervus/Nicolet file format (`read_nervus_header.m` and `read_nervus_data.m`) was the foundation for this Python port. Since then, we've added substantial GUID/event and channel ID parsing logic through our own reverse‑engineering work. Thank you to the FieldTrip team!

> **Note**: Some of our reverse‑engineered event labels are (unfortunately) in Norwegian.
>
> **Scope note**: This converter is primarily an in-house tool and includes some site-specific recovery heuristics (for example, fixed numeric-ID channel mappings used in our local recordings). These defaults improve our internal datasets, but may not match naming conventions used at other institutions.

## Quick Start

Clone the repository (or download it as a ZIP) from GitHub:

```bash
git clone https://github.com/haukurtg/e2edfconverter.git
cd e2edfconverter
```

The easiest way — no manual environment/dependency setup needed (`uv` handles it for you):

```bash
# Install uv if you don't have it (https://docs.astral.sh/uv/)
brew install uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Convert a single file
uv run --isolated nicolet-e2edf --in /path/to/recording.e --out ./edf_output

# Convert a folder of .e/.eeg files
uv run --isolated nicolet-e2edf --in ./my_eeg_folder --out ./edf_output
```

If you want a local environment for repeated use/development:

```bash
uv sync
uv run nicolet-e2edf --help
```

### Interactive Mode

For a guided experience with menus and progress bars:

```bash
uv run --isolated --with rich nicolet-e2edf --ui
```

![TUI Screenshot](docs/tui_screenshot.png)

## CLI Options

| Option | Description |
|--------|-------------|
| `--in` | Input `.e`/`.eeg` file or folder |
| `--out` | Output directory for EDF files |
| `--glob` | Filter pattern when input is a folder (e.g. `recording_*`) |
| `--json-sidecar` | Also emit a `.json` with metadata (channels, events, etc.) |
| `--split-by-segment` | Output one EDF per segment if the recording contains multiple segments |
| `--vendor-style` | Suppress system events to better match vendor EDF exports |
| `--resample-to` | Resample to a specific rate (Hz) (requires scipy) |
| `--lowcut` | High-pass filter cutoff in Hz (requires scipy) |
| `--highcut` | Low-pass filter cutoff in Hz (requires scipy) |
| `--notch` | Notch filter for powerline noise, e.g. `50` or `60` Hz (requires scipy) |
| `--ui` | Launch interactive terminal UI (requires rich) |
| `--verbose` | Show detailed logging |

**Filtering example:**

```bash
# Clinical defaults: 0.5–35 Hz bandpass + 50 Hz notch
uv run --isolated --with scipy nicolet-e2edf \
    --in ./data --out ./edf_output \
    --lowcut 0.5 --highcut 35 --notch 50
```

**Vendor-style comparison example:**

```bash
# Match vendor-style exports (split per segment + suppress system events)
uv run --isolated nicolet-e2edf \
    --in /path/to/recording.e --out ./edf_output \
    --split-by-segment --vendor-style --json-sidecar
```

## Viewing the Results

There's a bundled viewer script that shows your EDF in a double-banana montage:

```bash
uv run --isolated --with mne python inspect_edf.py ./edf_output/recording.edf
```

**Note:** When using the interactive TUI (`--ui`), the viewer is automatically launched with MNE in an isolated environment if needed. No manual installation required!

Options: `--lowcut`, `--highcut`, `--notch`, `--snapshot out.png` (for headless systems).

Filtering during conversion (`--lowcut`, `--highcut`, `--notch`) is lossy. In most cases, keep exports unfiltered and only use conversion-time filtering when you intentionally want a preprocessed output for direct downstream use (for example, an ML pipeline).

## Limitations

- Mixed sampling rates: default exports only dominant-rate channels; use `--resample-to` to include all "on" channels.
- When `--resample-to` is used, channels are resampled to the requested integer EDF rate.
- Events are written as EDF+ annotations
- EVENTTYPEINFOGUID labels are reverse-engineered; unknown GUIDs may be exported as UNKNOWN.
- `.eeg` support is currently not reliable; we need a larger `.eeg` dataset to implement and validate it properly.
- Some `.e` recordings store only numeric channel IDs (e.g., `1..64`). The numeric-channel fix and montage-recovery strategy (from `v0.2.5`) are mainly aimed at recovering channel names in atypical multi-channel EEG setups (`32`, `64`, `128`, etc.) using source montage derivations, fixed DERIVATION tables, and hidden montage catalogs.
- The CLI supports folder input, but processes files serially. For large cohorts, it is usually more efficient to call the CLI from a small batch wrapper that runs multiple workers and tracks progress/errors.

## Contributing

Contributions are welcome! If you're working on the EDF writer or want to understand the file format:

- **EDF+ Specification**: A copy of the full EDF+ specification is included at [`docs/EDF+ specification.pdf`](docs/EDF+%20specification.pdf). The official spec is also available at [edfplus.info](https://www.edfplus.info/specs/edfplus.html).
- **Tests**: Run `uv run pytest` to verify EDF+ compliance. We use PyEDFlib as a strict validator.

## License

GPL-3.0 — see `LICENSE`.

This project adapts logic from the [FieldTrip](https://github.com/fieldtrip/fieldtrip) toolbox (GPL-3.0).
