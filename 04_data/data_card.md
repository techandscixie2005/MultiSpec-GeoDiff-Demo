# Data Card: NIST IR Spectra

## File

- **Filename**: `IR_nist_200.jsonl`
- **Number of records**: 200
- **Modality**: Experimental IR

## Format

Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `smiles` | str | Ground-truth molecular structure in SMILES notation |
| `value.x` | list[float] | IR wavenumber axis (cm⁻¹) |
| `value.y` | list[float] | IR intensity (arbitrary units) |
| `temperature` | str or null | Measurement temperature |
| `pressure` | str or null | Measurement pressure |
| `condition` | str or null | Sample condition |
| `type` | str | Spectrum type (e.g., "experimental") |
| `source` | str | Data source attribution |

## Usage in Demo

- One molecule is selected by `--query_id` as the query spectrum.
- The remaining 199 molecules form the candidate library.
- Spectra are interpolated to a common grid (400–4000 cm⁻¹, 1800 points).
- Intensities are min-max normalized to [0, 1].
- Peaks are extracted for similarity matching.

This is a Stage-I retrieval and reranking demo. The small library is used to
demonstrate the pipeline, not to achieve production-level retrieval accuracy.

## Source

National Institute of Standards and Technology (NIST) Standard Reference
Database. The data has been reformatted for the demo pipeline; original data
is publicly available from NIST.

## Limitations

- **Small subset**: 200 molecules is a small library; retrieval accuracy is
  not benchmark-level.
- **IR only**: No Raman or other modalities included.
- **Not sufficient for training**: 200 spectra are not enough to train a full
  model. The demo uses a deterministic random encoder, not a trained neural
  network.
- **Demo-only usage**: This data is used exclusively for demonstrating the
  Stage-I pipeline, not for claims about real-world retrieval performance.
