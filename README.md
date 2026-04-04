# 📡 Radar-Based Vital Signs Detection Across Body Positions

> A real-time, non-contact vital signs monitoring system using an **IWR6843ISK 60GHz FMCW Radar Sensor**, combined with digital signal processing and machine learning to detect **Heart Rate** and **Respiration Rate** across different body positions and sensor placements.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-SocketIO-lightgrey)](https://flask.palletsprojects.com)
[![Radar](https://img.shields.io/badge/Sensor-IWR6843ISK-green)](https://www.ti.com/product/IWR6843)

---

## 🧠 Project Overview

This project investigates whether a **radar sensor** can reliably detect heart rate and respiration across **5 different body positions** (Supine, Sitting, Prone, Side, Standing) and **5 sensor placements** (Chest, Torso, Abdomen, Thigh, Calf).

The system combines:
- **FMCW Radar Physics** to extract micro-Doppler phase shifts from the skin surface
- **Butterworth Bandpass Filtering** to isolate cardiac (0.8–2.0 Hz) and respiratory (0.1–0.5 Hz) signals
- **Time-Domain Peak Detection** to calculate BPM in real-time
- **XGBoost ML Classifier** (Detectability) and **Random Forest Regressor** (HR Refinement) for intelligent post-processing

---

## 🗂️ Repository Structure

```
BodyPositionAnalysis/
│
├── realtime_server.py          # 🧠 Main Flask-SocketIO backend server
├── sensor_stream.py            # 📡 Radar communication & signal processing engine
│
├── frontend/
│   └── index.html              # 🖥️ Real-time dashboard UI (Chart.js + SocketIO)
│
├── config/
│   └── xwr68xx_profile_VitalSigns_20fps_Front.cfg  # ⚙️ Radar hardware configuration
│
├── position_aware_hr_model.joblib   # 🤖 Pre-trained Random Forest HR Regressor
├── detectability_classifier.joblib  # 🤖 Pre-trained XGBoost Detectability Classifier
│
├── train_position_aware_model.py    # Training script for HR regressor
├── train_detectability_classifier.py # Training script for detectability classifier
│
├── classifier_metrics.json     # Classifier performance metrics
├── model_metrics.json          # Regressor performance metrics
├── position_stats.json         # Per-position statistical summary
│
└── .gitignore
```

---

## ⚙️ Hardware Requirements

| Component | Specification |
|---|---|
| **Radar Sensor** | Texas Instruments IWR6843ISK (60–64 GHz FMCW) |
| **Carrier Board** | IWR6843ISK-ODS |
| **Control Port** | COM10 @ 115200 baud |
| **Data Port** | COM9 @ 921600 baud |
| **Firmware** | ES1.0 compatible SDK binary |

---

## 💻 Software Requirements

Install all Python dependencies:

```bash
pip install flask flask-socketio flask-cors pyserial numpy scipy scikit-learn xgboost joblib pandas
```

---

## 🚀 How to Run

### 1. Demo Mode (No Radar Required)
Run the server in demo mode with simulated data:
```bash
python realtime_server.py --demo
```
Then open your browser at:
```
http://localhost:5050
```

### 2. Live Radar Mode
Connect the IWR6843ISK sensor, then:
```bash
python realtime_server.py
```
In the dashboard, **uncheck "Demo Mode"** and click **Start**.

### 3. Upload a CSV Dataset
In the dashboard, upload one of your recorded `.csv` files and click **Start Demo** to replay it through the signal processing pipeline.

---

## 🔬 Signal Processing Pipeline

The system processes every radar frame through the following stages:

```
Raw I/Q ADC Samples (from Radar DSP)
        │
        ▼
  [1] Range-FFT (Hardware)
        │  → Converts raw samples to range bins (distance indexed)
        ▼
  [2] Bin Selection  →  target_bin = argmax(magnitude)
        │  → Locks onto the strongest reflection (the person)
        ▼
  [3] Phase Extraction  →  φ(t) = arctan2(Q, I)
        │  → Extracts micro-Doppler phase shift from skin movement
        ▼
  [4] Phase Unwrapping  →  np.unwrap(φ)
        │  → Removes 2π discontinuities for continuous displacement
        ▼
  [5] Detrending  →  scipy.signal.detrend()
        │  → Removes DC drift and slow ambient movement
        ▼
  [6] Butterworth Bandpass Filtering
        │  → Heart Wave:  0.8 – 2.0 Hz  (48–120 BPM)
        │  → Breath Wave: 0.1 – 0.5 Hz  (6–30 BPM)
        ▼
  [7] Peak Detection  →  scipy.signal.find_peaks()
        │  → Counts heartbeat / breath cycles in 10-second window
        ▼
  [8] BPM Calculation
        │  → BPM = (peak_count / window_duration_s) × 60
        ▼
  [9] ML Refinement (end of session)
        │  → Random Forest Regressor: refines HR based on posture
        │  → XGBoost Classifier: calculates Detectability confidence %
        ▼
  Dashboard Output (WebSocket → browser)
```

---

## 📊 Key Formulas

| Operation | Formula |
|---|---|
| Range Resolution | `range_m = bin_index × 0.044 m` |
| Phase Extraction | `φ(t) = arctan2(Q(t), I(t))` |
| Displacement | `Δd(t) = λ·Δφ(t) / 4π` |
| BPM from Peaks | `BPM = (N_peaks / T_window) × 60` |
| Nyquist Check | `Fs = 20 Hz >> 2 × 2.0 Hz (max cardiac freq)` |

---

## 🤖 Machine Learning Models

### 1. HR Regressor (`position_aware_hr_model.joblib`)
- **Type**: Random Forest Regressor
- **Input Features**: HeartRate_BPM, Posture, SensorPosition, Orientation_deg, Distance_m, SNR_dB, WaveEnergy
- **Output**: Refined Heart Rate (BPM) adjusted for body position

### 2. Detectability Classifier (`detectability_classifier.joblib`)
- **Type**: XGBoost Classifier
- **Input Features**: Posture, SensorPosition, Orientation_deg, Distance_m, SNR_dB, WaveEnergy
- **Output**: Probability (0–100%) that the current signal is a real physiological signal

---

## 📈 Experimental Datasets

Five datasets were recorded covering:

| Dataset | Posture | Sensor Position | Orientation |
|---|---|---|---|
| 01.csv | Lying Supine | Chest | 30° |
| 02.csv | Sitting | Torso Center | 0° |
| 03.csv | Lying Side | Abdomen | 90° |
| 04.csv | Lying Prone | Thigh | 45° |
| 05.csv | Standing | Calf | 60° |

---

## 🧪 Key Observations

- **Chest (Direct, 0°)** → Best signal quality. Highest SNR and Detectability.
- **Perpendicular (90°)** → Significant signal degradation. Phase shift amplitude drops.
- **Extremities (Thigh/Calf)** → Weakest signal. ML classifier required to separate signal from noise.
- **Respiration** is always stronger and more stable than Heart Rate in all positions.

---
