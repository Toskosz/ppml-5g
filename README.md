# PPML-5G: Privacy-Preserving Machine Learning for 5G/O-RAN Intrusion Detection

This project implements Fully Homomorphic Encryption (FHE)-based Random Forest classifiers for
network intrusion detection, using [Concrete ML](https://github.com/zama-ai/concrete-ml) by Zama.
Models are trained on public benchmark datasets and compiled to FHE circuits that can run inference
on encrypted traffic features — no plaintext data is ever exposed to the inference server.

---

## O-RAN Architecture Fit

These models target **IP-layer attack detection at the O-RAN data-plane edge** (N6/Gi interface —
the link between the UPF and the external data network). At this point, user-plane traffic is
standard IP flows and is fully compatible with CICFlowMeter-derived features.

```
UE --> gNB --> CU/DU --> UPF --> [N6 interface] --> Data Network
                                       ^
                              CICFlowMeter probe here
                              (syn_cnt, ack_cnt, tot_l_fw_pkt, ...)
                              Model runs FHE inference on flow features
```

**What these models cover:**
- DoS / DDoS attacks originating from UEs or targeting UEs
- Brute-force, botnet, web attacks traversing the N6 interface
- Any IP-layer attack visible as a CICFlowMeter flow record

**What these models do NOT cover (open research gap):**
- O-RAN control-plane attacks on the E2 interface (Near-RT RIC ↔ RAN nodes)
- Rogue base station detection (requires PHY/RRC-layer features)
- O1/A1 management-plane intrusions (YANG/NETCONF traffic)
- xApp authentication or RIC compromise

For RAN-layer anomaly detection, datasets from
[OpenRAN Gym / Colosseum](https://openrangym.com/datasets) (PHY/MAC KPIs from near-RT RIC) are
the relevant starting point — but no labeled attack benchmark exists for those interfaces yet.

---

## Model Families and Datasets

### 1. `cicids2018-kdd-models/` — Binary Classification

| Property | Value |
|---|---|
| **Dataset** | CSE-CIC-IDS-2018 |
| **Replaces** | KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated) |
| **Year** | 2018 |
| **Source** | Canadian Institute for Cybersecurity, hosted on AWS S3 |
| **Info page** | https://www.unb.ca/cic/datasets/ids-2018.html |
| **Task** | Binary classification: `0` = Benign, `1` = Attack |
| **Label column** | `label` (after `clean_col_names`); benign value = `'benign'` |
| **Numerical features** | `syn_cnt`, `ack_cnt`, `fin_cnt`, `rst_cnt`, `tot_l_fw_pkt` |
| **Categorical features** | `protocol`, `dst_port` |
| **Preprocessor** | `MinMaxScaler` + `OneHotEncoder` → `TruncatedSVD(100)` and `TruncatedSVD(200)` |
| **Saved preprocessor** | `preprocessor_cic_ids_2018.pkl` |
| **FHE model dirs** | `fhe_model_{n}_estimators_{d}_depth_svd_{100|200}_components/` |
| **Attack types** | FTP/SSH BruteForce, DoS (GoldenEye/Slowloris/Hulk), DDoS (LOIC/HOIC), Botnet, Web attacks, Infiltration |

**Download:**
```bash
aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
```

---

### 2. `cicunswnb15-models/` — Multi-class Classification

| Property | Value |
|---|---|
| **Dataset** | CIC-UNSW-NB15 (2024) |
| **Replaces** | NF-UNSW-NB15-v3 (UQ hosting returning 502; no confirmed v4 exists) |
| **Year** | 2024 (CIC re-extraction of UNSW-NB15 PCAPs using CICFlowMeter) |
| **Source** | Canadian Institute for Cybersecurity |
| **Info page** | https://www.unb.ca/cic/datasets/cic-unsw-nb15.html |
| **Task** | Multi-class classification (9 attack categories) |
| **Label column** | `Label` (original case; CIC-UNSW-NB15 preserves capital-L) |
| **Label values** | `Benign`, `Fuzzers`, `Analysis`, `Backdoor`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms` |
| **Numerical features** | `syn_cnt`, `ack_cnt`, `fin_cnt`, `rst_cnt`, `tot_l_fw_pkt` |
| **Categorical features** | `protocol`, `dst_port` |
| **Preprocessor** | `MinMaxScaler` + `OneHotEncoder` → `TruncatedSVD(100)` |
| **Saved preprocessor** | `preprocessor_cicunsw.pkl` |
| **Input file** | `CICFlowMeter_out.csv` |
| **FHE model dirs** | `fhe_model_{n}_estimators_{d}_depth_svd_100_components/` |

> **Note on feature changes from NF-UNSW-NB15-v3:** The previous NetFlow schema used
> `in_bytes`, `out_bytes`, `protocol`, `l7_proto`. CIC-UNSW-NB15 uses CICFlowMeter output,
> which does not produce `l7_proto`. The feature set was switched to the same CICFlowMeter
> schema used by the other two model families for consistency.

**Download:**
```bash
# Manual download from:
# https://www.unb.ca/cic/datasets/cic-unsw-nb15.html
# Direct: http://cicresearch.ca/CICDataset/CIC-UNSW/
wget -r -np -nd -A "*.csv" http://cicresearch.ca/CICDataset/CIC-UNSW/
```

---

## Shared Feature Schema (both model families)

Both model families now use the same CICFlowMeter-V3 feature set after migration to
CSE-CIC-IDS-2018 / CIC-UNSW-NB15:

| Feature | Type | Description |
|---|---|---|
| `syn_cnt` | Numerical | Count of packets with SYN flag |
| `ack_cnt` | Numerical | Count of packets with ACK flag |
| `fin_cnt` | Numerical | Count of packets with FIN flag |
| `rst_cnt` | Numerical | Count of packets with RST flag |
| `tot_l_fw_pkt` | Numerical | Total size (bytes) of packets in forward direction |
| `protocol` | Categorical | IP protocol number (TCP=6, UDP=17, ICMP=1, …) |
| `dst_port` | Categorical | Destination port number |

---

## Environment Setup

```bash
pip install -r requirements.txt
# macOS:
pip install -r macos_requirements.txt
```

---

## Running the Models

### Training (compiles FHE circuits)

```bash
# From the repo root:
python cicids2018-kdd-models/model.py
python cicunswnb15-models/model.py
```

Data files must be in the working directory as listed above. Each script will look for
a combined CSV cache before re-reading the folder.

### Single-record inference timing

```bash
python cicids2018-kdd-models/best_model_single_record.py
python cicunswnb15-models/single_record_inference.py
```

These scripts measure per-record plaintext and FHE inference latency over 1000 records.

---

## Dataset History

| Folder | Original Dataset | Year | Status | Current Dataset | Notes |
|---|---|---|---|---|---|
| `cicids2018-kdd-models/` | KDD Cup 1999 / NSL-KDD | 1999/2009 | Removed from CIC servers | CSE-CIC-IDS-2018 | Named for historical origin; now uses IDS-2018 |
| `cicunswnb15-models/` | NF-UNSW-NB15-v3 | 2021 | UQ hosting unavailable (502) | CIC-UNSW-NB15 (2024) | Renamed from `netflow-models/`; uses CICFlowMeter schema |

---

## Future Work

### Known Limitations / Pending Improvements

<!-- TODO: save processed train/test arrays to avoid re-preprocessing on every inference run -->
<!--
  Problem:
  cicids2018-kdd-models/best_model_single_record.py and cicunswnb15-models/single_record_inference.py
  both call load_and_preprocess(), which reads the full CSV, splits it, and applies the saved
  preprocessor + SVD on every run. This means inference startup time is as slow as training.

  Fix:
  After training (in model.py), serialize the final arrays:
    np.save('X_train_final_cic_ids_2018.npy', X_train_final)
    np.save('X_test_final_cic_ids_2018.npy', X_test_final)
    y_train_full.to_pickle('y_train_cic_ids_2018.pkl')
    y_test_full.to_pickle('y_test_cic_ids_2018.pkl')
    pd.Series(test_labels).to_pickle('test_labels_cic_ids_2018.pkl')  # string labels for printing

  Then in best_model_single_record.py, load_and_preprocess() should just np.load() those files
  instead of re-reading and re-splitting the CSV. Fall back to the full pipeline only if the
  .npy files are missing (first run).

  Same pattern applies to cicunswnb15-models/ with filenames *_cicunsw.npy / *.pkl.
-->

1. **Inference startup time** — `load_and_preprocess()` in both `cicids2018-kdd-models/best_model_single_record.py`
   and `cicunswnb15-models/single_record_inference.py` re-reads and re-splits the full CSV on every run.
   Training scripts should serialize the final `X_train`, `X_test`, `y_train`, `y_test` arrays
   (e.g. `np.save` / `pd.Series.to_pickle`) so inference scripts can load them directly instead of
   reprocessing from scratch.

---

### 5G-NIDD: O-RAN-Native IDS Dataset

<!-- TODO: implement fivegnidd-models/ -->
<!--
  Dataset: 5G-NIDD (5G Network Intrusion Detection Dataset)
  Source: VTT Technical Research Centre of Finland + Aalto University
  Year: 2022
  Availability: Public — IEEE DataPort
  URL: https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless
  DOI: 10.21227/s9er-2x95

  Why it matters for O-RAN:
  - Collected from a real 5G Standalone testbed using OpenAirInterface (OAI) + free5GC core
  - Contains actual 5G NR radio + core network traffic, not emulated LAN traffic
  - Attack types: DoS (UDP flood, TCP SYN), port scanning (TCP, UDP), fuzzing
  - Features: standard IP flow features (compatible with CICFlowMeter preprocessing)
  - Label: binary (benign/attack) and multi-class

  Implementation plan for fivegnidd-models/:
  1. Download dataset from IEEE DataPort (requires free account)
  2. Inspect column schema — likely compatible with CICFlowMeter feature names
  3. Create model.py mirroring cicunswnb15-models/model.py structure
  4. Create single_record_inference.py
  5. Document fit within O-RAN N6-interface context (same as other models)
  6. Note that this is the only dataset in the project collected from a real 5G testbed
-->

The closest existing dataset to true O-RAN-native IDS data is **5G-NIDD** (VTT/Aalto, 2022),
collected from a real 5G SA testbed using OpenAirInterface and free5GC. It represents actual
5G radio + core network traffic with labeled attack flows, making it the best current candidate
for validating these models in a genuine O-RAN context.

For **RAN-layer anomaly detection** (E2 interface, xApp telemetry, PHY/MAC KPIs), no public
labeled attack dataset currently exists. The
[Colosseum ColO-RAN dataset](https://github.com/wineslab/colosseum-oran-coloran-dataset)
provides near-RT RIC KPIs for scheduling optimization but contains no attack traffic.
Generating such a dataset requires a dedicated O-RAN testbed (e.g., Colosseum, POWDER/RENEW,
or an OAI + OpenAirInterface deployment) with synthetic attack injection.
