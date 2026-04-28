# FedMed

**FedMorph — Morphology-Aware Federated Learning for Medical Image Segmentation**

Anatomy-Decoupled Aggregation strategy for federated liver CT segmentation across distributed hospital sites.

## Architecture

![FedMed System Architecture](docs/architecture.png)

- **Model**: MONAI SegResNet + MorphologicalDescriptor + SegmentFeatureFusion (~1.2M params)
- **FL Framework**: [Flower](https://flower.ai/) (gRPC server–client)
- **Aggregation**: FedMorph — seg_head에만 quality-weighted, 나머지는 FedAvg

| Parameter Group | Aggregation Strategy |
|----------------|---------------------|
| Backbone (encoder/decoder) | Data-size weighted average (FedAvg) |
| GroupNorm layers | Data-size weighted average (FedAvg) |
| Segmentation head (`conv_final`) | Per-segment Dice quality × data-size weighted |
| Classification head + MorphDesc | Data-size weighted average (FedAvg) |

## Supported FL Methods

| Method | Description |
|--------|-------------|
| `FedAvg` | Standard weighted average |
| `FedProx` | FedAvg + proximal regularization term |
| `FedBN` | FedAvg with local normalization layers |
| `FedMorph` | Anatomy-Decoupled Aggregation (proposed) |

---

## Data Format

### CT 데이터 디렉토리 구조

각 환자 폴더 안에 `image.npy`와 `mask.npy` 두 파일이 필요합니다.
`patient.json` 등 별도 메타파일은 **필요 없습니다** — 데이터 폴더를 자동 스캔합니다.

```
combined/
├── patient_001/
│   ├── image.npy          # CT volume, shape: (D, H, W), dtype: float32 or uint8
│   └── mask.npy           # Segmentation mask, shape: (C, D, H, W), dtype: uint8
├── patient_002/
│   ├── image.npy
│   └── mask.npy
├── patient_003/
│   ├── image.npy
│   └── mask.npy
└── ...
```

> 폴더명이 곧 환자 ID입니다. 어떤 이름이든 상관없습니다.

**mask.npy 채널 구성** (C = num_classes + 1):
- `mask[0]`: background
- `mask[1]` ~ `mask[9]`: 9개 간 세그먼트 (seg1, seg2, seg3, seg4a, seg4b, seg5, seg6, seg7, seg8)

> 코드에서는 `mask[1:10]`만 사용합니다 (background 제외).

---

## 실행 가이드 (Step-by-Step)

### Step 0. 환경 설치 (모든 PC)

```bash
# Python 3.10+ 필요
git clone https://github.com/AISeedHub/FedMed.git
cd FedMed
uv sync
```

### Step 1. 데이터 분배 준비 (서버 PC에서 1회만)

`prepare_client_data.py`를 실행하여 환자를 클라이언트별로 나눕니다.
**데이터 폴더 경로만 지정하면 자동으로 환자를 스캔합니다.**

```bash
uv run python src/use_cases/liver_segmentation/prepare_client_data.py \
    --data-dir /path/to/combined \
    --n-clients 3 \
    --output-dir fl_clients
```

실행 결과 `fl_clients/` 폴더가 생성됩니다:

```
fl_clients/
├── client_0_patients.json     # Client 0에 할당된 train/val 환자 ID
├── client_1_patients.json     # Client 1에 할당된 train/val 환자 ID
├── client_2_patients.json     # Client 2에 할당된 train/val 환자 ID
└── test_patients.json         # 공통 테스트 환자 ID
```

각 파일의 포맷 (폴더명이 자동으로 환자 ID가 됨):

```json
{
  "train": ["patient_001", "patient_003", "patient_005", ...],
  "val": ["patient_010", "patient_012"]
}
```

### Step 2. 각 클라이언트 PC에 데이터 복사

각 Windows PC에 아래 두 항목을 복사합니다:

```
PC-A (Client 0):
  D:\data\combined\          ← CT 데이터 전체 (image.npy, mask.npy)
  D:\data\fl_clients\        ← fl_clients/ 폴더 전체

PC-B (Client 1):
  D:\data\combined\
  D:\data\fl_clients\

PC-C (Client 2):
  D:\data\combined\
  D:\data\fl_clients\
```

> **참고**: 각 클라이언트는 자기에게 할당된 `client_N_patients.json`만 읽지만,
> CT 데이터 폴더(`combined/`)는 전체가 있어야 합니다 (자기 환자 폴더에 접근해야 하므로).

### Step 3. 설정 파일 확인

`src/use_cases/liver_segmentation/configs/base.yaml`:

```yaml
# Server
server_address: "0.0.0.0:9000"    # 서버 바인드 주소 (변경 불필요)
fl_rounds: 50                      # 연합학습 라운드 수
min_clients: 3                     # 최소 클라이언트 수 (모두 접속해야 시작)
local_epochs: 10                   # 라운드당 로컬 학습 에포크 수

# Method
method: "FedMorph"                 # FedAvg | FedProx | FedBN | FedMorph

# Training
batch_size: 1
learning_rate: 3.0e-4
morph_coeff: 0.005                 # 후반 라운드에서만 점진 적용 (0→0.005)
cls_coeff: 0.0                     # 비활성

# Data paths (기본값, 각 PC에서 환경변수/인자로 오버라이드)
data_dir: "/data/jin/data/combined"
client_data_dir: "/data/jin/FedFace/fl_clients"
```

### Step 4. 서버 실행 (서버 PC)

서버를 **먼저** 실행합니다. 모든 클라이언트가 접속할 때까지 대기합니다.

```bash
# Linux
./src/run_liver_server.sh

# Windows
src\run_liver_server.bat
```

또는 직접:

```bash
uv run python src/use_cases/liver_segmentation/main_server.py \
    --config src/use_cases/liver_segmentation/configs/base.yaml
```

서버가 시작되면 아래와 같은 메시지가 출력됩니다:

```
FedMorph - Liver Segmentation Server
============================================================
  Method:       FedMorph
  Rounds:       50
  Min Clients:  3
  Local Epochs: 10
============================================================
Listening on 0.0.0.0:9000
Waiting for 3 clients to connect ...
```

### Step 5. 클라이언트 실행 (각 PC)

서버가 실행된 후, 각 클라이언트 PC에서 실행합니다. **순서는 상관없습니다.**

```bash
# Windows — 서버 IP가 192.168.1.100인 경우
src\run_liver_client.bat 0 192.168.1.100:9000     # PC-A (Client 0)
src\run_liver_client.bat 1 192.168.1.100:9000     # PC-B (Client 1)
src\run_liver_client.bat 2 192.168.1.100:9000     # PC-C (Client 2)

# Linux
./src/run_liver_client.sh 0 192.168.1.100:9000
```

또는 직접:

```bash
uv run python src/use_cases/liver_segmentation/main_client.py \
    --client-id 0 \
    --server-address 192.168.1.100:9000 \
    --data-dir D:\data\combined \
    --client-data-dir D:\data\fl_clients
```

**데이터 경로 오버라이드 방법 (우선순위 높은 순):**

| 방법 | 예시 |
|------|------|
| 커맨드라인 인자 | `--data-dir D:\data\combined` |
| 환경변수 | `set FEDMORPH_DATA_DIR=D:\data\combined` |
| config YAML | `data_dir: "D:\data\combined"` |

### Step 6. 학습 진행

3개 클라이언트가 모두 접속하면 자동으로 학습이 시작됩니다.

```
[Client 0] === Round 1 ===
[Client 0] Epoch 3/10, Loss: 1.2345, LR: 0.000300
[Client 0] Epoch 6/10, Loss: 0.9876, LR: 0.000300
[Client 0] Epoch 10/10, Loss: 0.8765, LR: 0.000300
[Client 0] Seg Dice mean: 0.1234, Morph div: 0.000456
```

---

## 네트워크 요구사항

| 항목 | 서버 PC | 클라이언트 PC |
|------|---------|--------------|
| 고정 IP | **필요** (또는 DDNS) | 불필요 |
| 포트 개방 | **9000번 인바운드** 허용 | 불필요 |
| 방화벽 | 9000 포트 허용 설정 | 별도 설정 없음 |

**Windows 방화벽 포트 개방 (서버 PC만):**

```powershell
netsh advfirewall firewall add rule name="FedMorph Server" dir=in action=allow protocol=TCP localport=9000
```

> 클라이언트는 서버로 **아웃바운드** TCP 연결만 하므로 포트 개방이 필요 없습니다.

---

## 실행 순서 요약

```
┌─────────────────────────────────────────────────────────────┐
│  1. prepare_client_data.py  (서버 PC에서 1회)               │
│     → fl_clients/ 생성                                      │
│                                                             │
│  2. 데이터 복사                                              │
│     → combined/ + fl_clients/ 를 각 PC에 복사               │
│                                                             │
│  3. 서버 실행  (서버 PC)                                     │
│     → run_liver_server.bat                                  │
│     → "Waiting for 3 clients..." 메시지 확인                 │
│                                                             │
│  4. 클라이언트 실행  (각 PC, 순서 무관)                       │
│     → run_liver_client.bat 0 <SERVER_IP>:9000               │
│     → run_liver_client.bat 1 <SERVER_IP>:9000               │
│     → run_liver_client.bat 2 <SERVER_IP>:9000               │
│                                                             │
│  5. 3개 모두 접속 → 자동 학습 시작 (50 rounds)               │
└─────────────────────────────────────────────────────────────┘
```

---

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
# Linux
export FEDMORPH_DATA_DIR=/path/to/combined
export FEDMORPH_CLIENT_DATA_DIR=/path/to/fl_clients

# Windows
set FEDMORPH_DATA_DIR=D:\data\combined
set FEDMORPH_CLIENT_DATA_DIR=D:\data\fl_clients
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
