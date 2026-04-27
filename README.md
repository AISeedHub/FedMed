# FedMed

**FedMorph — Morphology-Aware Federated Learning for Medical Image Segmentation**

Anatomy-Decoupled Aggregation strategy for federated liver CT segmentation across distributed hospital sites.

## Architecture

- **Model**: MONAI SegResNet + MorphologicalDescriptor + SegmentFeatureFusion (~1.2M params)
- **FL Framework**: [Flower](https://flower.ai/) (gRPC server–client)
- **Aggregation**: FedMorph — 4 parameter groups, each with a tailored strategy:

| Parameter Group | Aggregation Strategy |
|----------------|---------------------|
| Backbone (encoder/decoder) | Data-size weighted average (FedAvg) |
| GroupNorm layers | Not aggregated (stays local per client) |
| Segmentation head (`conv_final`) | Per-segment Dice quality-weighted |
| Classification head + MorphDesc | Morphological-diversity weighted |

## Supported FL Methods

| Method | Description |
|--------|-------------|
| `FedAvg` | Standard weighted average |
| `FedProx` | FedAvg + proximal regularization term |
| `FedBN` | FedAvg with local normalization layers |
| `FedMorph` | Anatomy-Decoupled Aggregation (proposed) |

## Quick Start

### 1. Install

```bash
uv sync
```

### 2. Prepare Client Data

```bash
uv run python src/use_cases/liver_segmentation/prepare_client_data.py \
    --patient-json /path/to/patient.json \
    --n-clients 3 \
    --output-dir fl_clients
```

### 3. Run Server

```bash
# Linux
./src/run_liver_server.sh

# Windows
run_liver_server.bat
```

### 4. Run Clients (on each PC)

```bash
# Linux
./src/run_liver_client.sh 0 <SERVER_IP>:9000

# Windows
run_liver_client.bat 0 <SERVER_IP>:9000
```

## Configuration

Edit `src/use_cases/liver_segmentation/configs/base.yaml`:

```yaml
method: "FedMorph"        # FedAvg | FedProx | FedBN | FedMorph
fl_rounds: 50
min_clients: 3
local_epochs: 10
```

Per-machine data paths can be overridden via environment variables:

```bash
export FEDMORPH_DATA_DIR=/path/to/combined
export FEDMORPH_CLIENT_DATA_DIR=/path/to/fl_clients
```

## Project Structure

```
src/
  fed_core/
    fed_server.py            # Flower server wrapper
    fed_client.py            # Abstract FL client base
    fedmorph_strategy.py     # FedMorph aggregation strategy
  use_cases/liver_segmentation/
    configs/base.yaml        # Training & FL configuration
    models/
      segresnet_cirrhosis.py # SegResNet + MorphDesc + SegmentFeatureFusion
    utils/
      dataset.py             # 9-segment liver CT dataset
      loss.py                # Seg + Cls + Morph consistency loss
      metrics.py             # Dice / HD95 / AUC evaluation
    main_server.py           # Server entry point
    main_client.py           # Client entry point
    prepare_client_data.py   # Data distribution script
  run_liver_server.bat/.sh   # Server launch scripts
  run_liver_client.bat/.sh   # Client launch scripts
```

## Acknowledgments

Built on the [AISeedHub/FedFace](https://github.com/AISeedHub/FedFace) federated learning framework.
