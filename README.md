# FedMEvo

**Joint Agentic Memory Evolution and Communication Optimization for Wireless Federated Agent Collaboration**

FedMEvo jointly optimizes agentic memory evolution and communication for wireless
federated agent collaboration. This project implements memory construction,
multi-granularity representation, demand- and novelty-aware utility, joint
scheduling, selective payload upload, personalized delivery, and memory reuse.

This project has **two parts**:

| Part | Location | Purpose |
| --- | --- | --- |
| Core package | `src/fedmevo/` | Reusable memory evolution and communication optimization |
| Industrial integration | `industrial/` | Industrial application integration and the CoalMine Methane-Risk Identification Dataset |

## 1. Install and run the core

Python 3.10+ is required. The core and HTTP backend use the standard library.
From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell, instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python examples/run_demo.py --backend demo
```

The last command writes `outputs/demo.json`: selected memory IDs, semantic
levels, allocated bandwidth, round-wise communication, recipient libraries,
and predictions. `DemoBackend` is an explicit deterministic **wiring fixture**,
not an LLM. Its fixed answers are not performance evidence.

## 2. Run the industrial integration

```bash
python -m industrial.run_coalmine --backend demo
```

This runs the CoalMine Methane-Risk Identification Dataset with five clients,
two rounds, and two validation targets per client. Demo predictions are not scored.

See [industrial/README.md](industrial/README.md) for data fields, the original
split, API/local examples, and adaptation to another industrial application.

## 3. Connect an API-hosted LLM

The API backend expects a chat-completions-compatible endpoint accepting
`model`, `messages`, `temperature`, `max_tokens`, and `stream`, and returning
`choices[0].message.content`. Set the service root including `/v1`; the client
appends `/chat/completions`.

Set the key in your shell without placing it in source or config files:

```bash
# Linux/macOS: replace the placeholder locally.
export FEDMEVO_API_KEY="YOUR_KEY"
python examples/run_demo.py --backend api \
  --base-url https://YOUR-SERVICE/v1 --model YOUR-MODEL
python -m industrial.run_coalmine --backend api \
  --base-url https://YOUR-SERVICE/v1 --model YOUR-MODEL
```

PowerShell:

```powershell
$env:FEDMEVO_API_KEY = "YOUR_KEY"
python -m industrial.run_coalmine --backend api --base-url https://YOUR-SERVICE/v1 --model YOUR-MODEL
```

No hosted calls happen unless `--backend api` is explicitly selected. For the
default industrial run, up to 60 generation requests are made: 10 candidate
distillations, 40 local replay predictions, and 10 final predictions. Earlier
stopping may reduce this count. Provider fees and token limits apply. Input
observations, historical feedback, and selected memories are sent to the
configured service. Only use data you are permitted to send.

Distillation expects a plain JSON response with nonempty `core`, `conditions`,
and `procedure` strings. The API backend generates answers from task observations
and retrieved memories.

## 4. Connect a locally loaded model

Install a PyTorch build appropriate for your hardware, then:

```bash
python -m pip install -e ".[local]"
python examples/run_demo.py --backend local --model /path/to/text-instruct-model --device cuda
python -m industrial.run_coalmine --backend local --model /path/to/text-instruct-model --device cuda
```

On Windows the model argument can be a quoted Windows directory. Use
`--device cpu` for CPU inference. `--max-tokens` defaults to 512.

The local backend loads a complete, already-downloaded **text causal-LM**
checkpoint and tokenizer, uses its chat template, freezes all weights, and
generates without sampling. CPU uses float32 and CUDA uses float16. The model
must fit on the selected device; no quantization or multi-GPU sharding is added.
No checkpoint is bundled or automatically downloaded. Qwen2.5-VL-style multimodal
loading is not implemented by this text-only loader; do not use a VL checkpoint
with it. A text instruct checkpoint supported by `AutoModelForCausalLM` is needed.

The implementation follows the [Transformers model loading interface](https://huggingface.co/docs/transformers/models).

## 5. How the core works

1. A client distills completed local tasks and feedback into reusable memories.
2. Core, conditions, and procedure form compact, standard, and detailed payloads.
3. Local replay estimates utility; a quantized descriptor exposes utility, size,
   capability tag, and a preview vector, not the memory text.
4. Peer demand and novelty adjust utility; per-level top-k menus are built.
5. The scheduler jointly chooses memory count and level. For a fixed selection,
   bandwidth is proportional to `sqrt(payload_bits / spectral_efficiency)`.
6. Only granted payloads enter the shared pool. Fusion and recipient-specific
   coverage checks produce serialized updates.
7. Received memories are retrieved for subsequent construction and inference.
   Execution stops on exhausted novelty, exhausted experience, or the round limit.

## 6. CoalMine Methane-Risk Identification Dataset

We introduce the **CoalMine Methane-Risk Identification Dataset** for studying
federated agent collaboration in underground industrial IoT environments. We
transform real-world, multi-sensor time series into contextual risk-identification
tasks that connect local observations, historical experience, and future methane
events. The dataset has undergone data checks and expert evaluation. It supports
memory construction, cross-client memory exchange, and evaluation of subsequent
risk predictions.

The underlying Methane dataset contains **9,199,930 one-second sensor records**.
Our task observations summarize methane
concentrations, ventilation and airflow, temperature, humidity, pressure, and
shearer operating conditions.

Download the [single-file dataset](industrial/data/coalmine_methane.json).
Each record includes its task ID, client assignment, timestamp, semantic tag,
observation-based question, answer choices, and label. Construction records also
preserve their construction roles.

The original sensor data are credited to **Marek Sikora and Łukasz Wróbel**,
*Methane*, Mendeley Data, V1 (2021),
[doi:10.17632/yd7vw4c5mk.1](https://doi.org/10.17632/yd7vw4c5mk.1), under
**CC BY 4.0**. See [data attribution and license](industrial/DATA_LICENSE.md).
