# Netslab-5G-ORAN-IDD — Dataset Report

**Status:** documentation only — data is not in this repo and has not been
downloaded to the training machine yet.

| Field | Value |
|---|---|
| **Full name** | NetsLab-5GORAN-IDD (Kaggle: Netslab-5G-ORAN-IDD) |
| **Publisher** | Network Softwarization and Security Labs (NetsLab) |
| **Kaggle** | https://www.kaggle.com/datasets/netslabdemo/netslab-5g-oran-idd |
| **License** | CC BY 4.0 |
| **Size (Kaggle)** | ~145.96 GB, 320 files |
| **Usability (Kaggle)** | 5.29 |
| **Last checked** | 2026-11-07 |

## Citation

Civciss, A., Ravihansa, V., Abed Zadeh, F., Sandeepa, C. & Liyanage, M., 2025.
*Silent Signals, Loud Threats: Using dApps for Radio Signal Intelligence-based
Intrusion Detection in 5G O-RAN.* In: Proceedings of the 2025 IEEE Global
Communications Conference: Communication & Information Systems Security, IEEE.

If this dataset is used in experiments, cite the paper above.

## Purpose

Labeled intrusion-detection data from a real **OpenAirInterface (OAI) 5G O-RAN
testbed**, with **two synchronized views** of the same traffic:

1. **Network / higher layer** — packet captures and Zeek flow features from the
   O-CU (IP/application-visible traffic).
2. **Radio / lower layer** — physical and MAC telemetry from the O-DU, exposed
   toward the Near-RT RIC path over **E2** (OAIbox).

The paper framing is radio-signal intelligence via **dApps** on the Near-RT
RIC. The network-layer captures are a parallel upper-layer view, not an E2
stream.

This is relevant to the PPML-5G README gap around RAN-layer / E2-visible
detection (PHY/MAC KPIs), as opposed to the existing CICFlowMeter N6-style
pipelines in `cicids2018-kdd-models/` and `cicunswnb15-models/`.

## Testbed

| Component | Count / notes |
|---|---|
| Core platform | OpenAirInterface (OAI) 5G O-RAN |
| O-CU | 1 |
| O-DU | 2 |
| O-RU | 1 |
| O-Cloud / Near-RT RIC host | Dell Precision 7920, co-located with Near-RT RIC |
| UE | 2 physical devices (benign clients and attack sources) |

## How data reaches the RIC (important)

**Only the lower (radio) layer is described as going over E2.**

| Layer | Capture point | Path to RIC / collector | Raw files | Aggregated summary |
|---|---|---|---|---|
| **Lower (radio)** | O-DU via OAIbox | **E2 interface** (Near-RT RIC / dApp path) | `.txt` under `*/lower_layer/` | `Lower_Layer_Data.db` (~5.4 MB) |
| **Network (higher)** | O-CU | **Wireshark** at O-CU (not E2); Zeek feature extraction | `.pcap` under `*/network_layer/` | `Network_Dataset.csv` and `Network_Dataset.db` |

Do not assume both layers are E2 SM / KPM exports. Network data is a separate
CU-side PCAP pipeline.

## Top-level layout (Kaggle Data Explorer)

```
Attack (Legacy) Version 1/     # older high-volume flood captures
Benign/
BruteForce/
DDOS/                          # spelling on Kaggle tree (card also says DDoS)
DoS/
Probe/
Web/
Lower_Layer_Data.db            # combined radio telemetry
Network_Dataset.csv            # combined network/flow table
Network_Dataset.db             # same network summary as DB
```

### Version 2 category folders

Each of `Benign`, `DoS`, `DDOS`, `Probe`, `BruteForce`, `Web` contains:

```
<Category>/
  lower_layer/     # radio telemetry .txt
  network_layer/   # Wireshark .pcap
```

Filenames encode attack type and tool, for example:

- `DoS_SYN_nping3.pcap`
- `portscan_masscan.txt`

### Version 1 (legacy)

Top-level folder `Attack (Legacy) Version 1/`:

- Focus: high-volume flooding (DoS/DDoS SYN, TCP, ACK, ICMP)
- Limited tool diversity
- Prefer Version 2 directories for broader attack coverage unless a study
  specifically needs the legacy floods

## Traffic and labels

### Benign

YouTube streaming, file downloads, browsing, system updates, controlled TCP/UDP
via iPerf3.

### Attack categories (Version 2)

| Category | Subtypes (as documented) |
|---|---|
| **DoS** | SYN, ACK, UDP, ICMP, HTTP floods |
| **DDoS** | Same flood families, multi-UE |
| **Probe** | Port scanning, OS fingerprinting |
| **BruteForce** | SSH login, FTP login |
| **Web** | HTTP brute force, directory brute force, SQL injection, XSS |

### Attack tools (documented)

| Family | Tools |
|---|---|
| DoS / DDoS | nping, hping3, Slowloris.py |
| Probe | Nmap, Masscan, Xprobe2 |
| Brute force | Hydra, Medusa |
| Web | Gobuster, Sqlmap, DVWA, XSStrike |

Version 2 varies stealth, packet rate, concurrency, and protocol selection
(example: Probe with Masscan / Nmap / Xprobe2 under different timing templates,
retry counts, and packet types).

## Feature content (as documented)

Exact column schemas were **not** verified from the files in this workspace
(CSV/DB inspection was not completed here). The Data Card states:

### Network layer (Zeek-derived examples)

Source logs: `conn.log`, `http.log`, `files.log`.

Example fields: `src_bytes`, `protocol`, `duration`, `http_status_error`,
`is_file_transferred`.

### Radio / lower layer

- **22 physical/MAC metrics per record**
- Examples: RSRP, CQI, RSSI, SINR, MCS, Block Error Rate

Kaggle summary also lists **2106 columns** across the dataset explorer view —
treat that as an aggregate explorer statistic until `Network_Dataset.csv` is
opened locally and column lists are confirmed.

## Ready-to-use vs raw bulk

| Artifact | Role | Practical note |
|---|---|---|
| `Network_Dataset.csv` / `.db` | Combined upper-layer table | Likely first stop for tabular ML |
| `Lower_Layer_Data.db` | Combined radio telemetry | SQLite; Kaggle shows no CSV preview |
| Per-class `network_layer/*.pcap` | Raw traffic | Dominates the ~146 GB footprint |
| Per-class `lower_layer/*.txt` | Per-scenario radio logs | Pairable with matching pcaps by naming |

For PPML-style training, prefer the summary tables first; only pull PCAPs if
re-extraction or packet-level work is required.

## Relation to existing PPML-5G datasets

| | CIC-IDS-2018 / CIC-UNSW-NB15 (current models) | Netslab-5G-ORAN-IDD |
|---|---|---|
| Domain | Generic / enterprise IDS PCAPs re-flowed | Live OAI 5G O-RAN testbed |
| Feature schema | CICFlowMeter compact flow fields | Zeek network features + 22 radio KPIs |
| O-RAN interface | Conceptual N6/edge IP flows | Radio path via **E2**; network via **O-CU PCAP** |
| Drop-in for current trainers | Yes (after download + existing loaders) | **No** — needs new preprocessor / loaders |
| Size | Multi-GB CSV suites | ~146 GB (mostly raw) |

Current Concrete ML scripts expect CICFlowMeter-style columns
(`syn_flag_cnt`, `ack_flag_cnt`, …, `protocol`, `dst_port`). Netslab network
features are Zeek-oriented and the radio table is a different modality.

## Download

```bash
# Requires a Kaggle account + API token (~/.kaggle/kaggle.json)
kaggle datasets download -d netslabdemo/netslab-5g-oran-idd \
  -p netslab-5g-oran-idd/data --unzip
```

Expect large disk use (~146 GB). Prefer selective download of
`Network_Dataset.csv`, `Network_Dataset.db`, and `Lower_Layer_Data.db` if the
Kaggle CLI / UI allows file-level pulls, before fetching all PCAPs.

## Open follow-ups (before training)

1. Inspect `Network_Dataset.csv` header, label column name/values, and class
   balance.
2. Inspect `Lower_Layer_Data.db` tables/schema and confirm the 22 radio metrics.
3. Decide task: radio-only, network-only, or multimodal fusion.
4. Decide binary (benign vs attack) vs multi-class (six Version 2 categories).
5. Design a split that avoids leakage across paired lower/network files from the
   same capture scenario.
6. Only then sketch a Concrete ML preprocessor (do not reuse CIC SVD pipelines
   unchanged).

## Sources for this report

- Kaggle Data Card and Data Explorer for
  `netslabdemo/netslab-5g-oran-idd` (inspected 2026-11-07).
- Dataset structure and collection wording as published on that page; local file
  contents were not verified in this repository.
