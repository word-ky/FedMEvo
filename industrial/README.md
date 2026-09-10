# Industrial integration: CoalMine

This directory provides industrial application integration for FedMEvo, using the
CoalMine Methane-Risk Identification Dataset. It supports both API-hosted LLMs
and locally loaded models.

```text
CoalMine observation/task
  -> adapter.model_task
  -> FedMEvo memory evolution and scheduling
  -> APIBackend or LocalBackend
  -> prediction and post-inference scoring
```

## CoalMine Methane-Risk Identification Dataset

We construct contextual methane-risk identification tasks from real-world mine
sensor time series for federated agent collaboration. The dataset has undergone
data checks and expert evaluation. Observations cover methane,
ventilation and airflow, temperature, humidity, pressure, and shearer operation.
See the [dataset overview](../README.md#6-coalmine-methane-risk-identification-dataset)
for the task design and original source attribution.

`data/coalmine_methane.json` is a **single self-contained JSON file** containing:

- Source attribution, CC BY 4.0 link, transformation note, and original artifact hashes.
- 250 main task samples, with IDs, questions, choices, timestamps,
  client assignments, semantic tags, and answer labels.
- Main task splits: 60 construction, 40 validation, 150 test.

Each record has `id`, `split`, `client_id`, `timestamp`, `semantic_tag`,
`question`, `choices`, and `answer`. Construction records also have
`construction_role`. `A` means Normal; `B` means Warning.

The original sensor source contains 9,199,930 one-second records. Construction,
validation, and test splits each contain equal numbers of Normal and Warning
samples.

## Run from the repository root

```bash
python -m industrial.run_coalmine --backend demo
```

Default wireless parameters are
500 Hz total bandwidth, latency weight 0.005, demand weight 0.4, and deterministic
example spectral efficiencies 2/3/4/5/6 bits/s/Hz.

API-hosted model (set `FEDMEVO_API_KEY` first):

```bash
python -m industrial.run_coalmine --backend api --base-url https://YOUR-SERVICE/v1 --model YOUR-MODEL
```

Complete local text model (install `.[local]` first):

```bash
python -m industrial.run_coalmine --backend local --model /path/to/text-instruct-model --device cuda
```

For all 150 test targets, select:

```bash
python -m industrial.run_coalmine --backend local --model /path/to/text-instruct-model --device cuda --split test --targets-per-client 30
```

The default evaluation split is validation.
API compatibility, costs, local loader restrictions, and setup are documented in
the [main README](../README.md). The demo backend has no meaningful accuracy;
real backends report post-inference accuracy for the requested subset only.

## Adapt another industrial task

Adapt `adapter.py` to convert your application's observations and historical
feedback into task inputs. Configure the client assignments and evaluation
targets, then select `--backend api` or `--backend local` for LLM inference.
