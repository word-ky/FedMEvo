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

## Experimental configuration

The complete configuration used for the federated agentic methane-risk
identification experiments is summarized below.

| Setting | Configuration |
| --- | --- |
| **System Deployment** | |
| Central host | CPU-based orchestration; 2 × Intel Xeon Platinum 8358P @ 2.60 GHz; 64 physical cores / 128 threads |
| Edge-node platform | GPU-based local inference; 2 × NVIDIA RTX A6000, 48 GB/GPU |
| Number of clients | 5 logical clients |
| Default communication rounds | 2 rounds |
| Default edge-node model | Qwen2.5-VL-7B-Instruct, frozen text-only inference |
| Heterogeneous model assignment | Clients 0 and 1: Qwen2.5-VL-7B; clients 2 and 3: Qwen3-8B; client 4: Llama-3.1-8B-Instruct |
| Inference mode | Locally loaded frozen models; text-only input; deterministic generation without sampling |
| Model precision | BF16 |
| Inference concurrency | One inference request per backend at a time |
| Qwen3 inference mode | Thinking disabled; non-thinking mode |
| **Communication Configuration** | |
| Shared uplink bandwidth | 500 Hz |
| Client spectral efficiencies | 1.8 / 1.2 / 0.5 / 0.9 / 1.5 bit/s/Hz for clients 0–4, respectively |
| Network model | Reliable transmission with zero packet loss |
| Communication accounting | Uplink, recipient-charged downlink, control, and total bytes |
| **Data Configuration** | |
| Source stream | 9,199,930 one-second underground-mine multisensor records |
| Observation / horizon / lead | 60 s / 180 s / 180 s |
| Risk criterion | Methane concentration ≥ 1.0% across monitored sensors |
| Task type | Binary methane-risk identification: Normal vs. Warning |
| Data split | 60 construction / 40 validation / 150 test |
| Class distribution | Construction: 30 Normal / 30 Warning; validation: 20 Normal / 20 Warning; test: 75 Normal / 75 Warning |
| Input features | Textual summaries of methane concentration, ventilation and airflow, temperature and humidity, pressure, and shearer operating status |
| **Evaluation** | |
| Evaluation metrics | Accuracy; Warning-class recall and Normal-class specificity are additionally reported |

## Run from the repository root

```bash
python -m industrial.run_coalmine --backend demo
```

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
