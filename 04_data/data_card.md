# Data Card: NIST IR Spectra

## File

- `IR_nist_200.jsonl` — 200 experimental IR spectra from the NIST database.

## Format

Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `smiles` | str | Ground-truth molecular structure in SMILES notation |
| `value.x` | list[float] | IR wavenumber axis (cm^-1) |
| `value.y` | list[float] | IR intensity (arbitrary units) |
| `temperature` | str or null | Measurement temperature |
| `pressure` | str or null | Measurement pressure |
| `condition` | str or null | Sample condition |
| `type` | str | Spectrum type (e.g., "experimental") |
| `source` | str | Data source attribution |

## Usage in Demo

- One molecule is selected by `--query_id` as the query spectrum.
- The remaining 199 molecules form the candidate library.
- Spectra are interpolated to a common grid (400–4000 cm^-1, 1800 points).
- Intensities are min-max normalized to [0, 1].
- Peaks are extracted for similarity matching.

## Source

National Institute of Standards and Technology (NIST) Standard Reference Database.
The data has been reformatted for the demo pipeline; original data is publicly
available from NIST.

## Limitations

- 200 molecules is a small library; retrieval accuracy is not benchmark-level.
- The dataset may have biases in molecular diversity.
- Experimental noise and variable resolution affect cross-spectrum comparison.
- This data is used exclusively for demonstrating the Stage-I pipeline,
  not for claims about real-world retrieval performance.
