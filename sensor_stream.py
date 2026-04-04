"""
sensor_stream.py — Real-Time Vital Signs Streamer for IWR6843ISK

Reads TLV packets from the radar sensor via serial ports and prints
one JSON line per frame to stdout. The parent process (realtime_server.py)
reads these lines and pushes them to the frontend via WebSocket.

Usage:
  python sensor_stream.py                          # Real sensor mode
  python sensor_stream.py --demo                   # Simulated data mode
  python sensor_stream.py --port COM5 --dport COM6 # Custom ports
"""

import serial
import time
import struct
import statistics
import os
import sys
import json
import argparse
import math
import random
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks

# ────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="IWR6843ISK Vital Signs Streamer")
parser.add_argument("--demo", action="store_true", help="Run with simulated data")
parser.add_argument("--file", type=str, help="Stream data from a synthetic CSV file")
parser.add_argument("--port", default="COM10", help="User/config serial port")
parser.add_argument("--dport", default="COM9", help="Data serial port")
parser.add_argument("--duration", type=int, default=30, help="Recording duration (seconds)")
parser.add_argument("--cfg", default=os.path.join("config", "xwr68xx_profile_VitalSigns_20fps_Front.cfg"),
                    help="Path to radar config file")
args = parser.parse_args()

DEMO_MODE = args.demo
FILE_MODE = args.file
USER_PORT = args.port
DATA_PORT = args.dport
DURATION = args.duration
CFG_FILE = args.cfg

FPS = 20
BUF_LEN = 200
MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def bandpass(sig, low, high, fs, order=4):
    if len(sig) < fs * 2:
        return sig
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")

def normalize(x):
    s = np.std(x)
    return x if s == 0 else x / s

def extract_range_m(tlv):
    for offset in range(64, min(128, len(tlv)), 4):
        try:
            val = struct.unpack_from("<f", tlv, offset)[0]
            if 0.2 < val < 3.0:
                return val
        except:
            pass
    return None

def emit(obj):
    """Print a JSON line to stdout for the parent process to read."""
    print(json.dumps(obj), flush=True)

# ────────────────────────────────────────────────────────────
# DEMO MODE — Simulated Vital Signs
# ────────────────────────────────────────────────────────────
def run_demo():
    emit({"type": "status", "message": "DEMO MODE — Generating simulated vital signs"})

    base_hr = random.uniform(65, 85)
    base_rr = random.uniform(14, 20)
    start = time.time()

    hr_history = []
    rr_history = []
    heart_phase = []
    breath_phase = []

    frame_idx = 0
    while time.time() - start < DURATION:
        ts = time.time() - start
        t = frame_idx / FPS

        # Simulate realistic HR and RR with slight drift
        hr = base_hr + 3 * math.sin(0.1 * t) + random.gauss(0, 0.5)
        rr = base_rr + 1.5 * math.sin(0.07 * t) + random.gauss(0, 0.3)
        hr = max(50, min(120, hr))
        rr = max(8, min(30, rr))

        # Simulate waveforms
        heart_wave = 0.8 * math.sin(2 * math.pi * (hr / 60) * t) + random.gauss(0, 0.05)
        breath_wave = 1.2 * math.sin(2 * math.pi * (rr / 60) * t) + random.gauss(0, 0.03)
        range_m = 0.7 + 0.02 * math.sin(0.5 * t) + random.gauss(0, 0.005)

        hr_history.append(hr)
        rr_history.append(rr)
        heart_phase.append(heart_wave)
        breath_phase.append(breath_wave)

        emit({
            "type": "frame",
            "ts": round(ts, 3),
            "hr": round(hr, 1),
            "rr": round(rr, 1),
            "heart_wave": round(heart_wave, 4),
            "breath_wave": round(breath_wave, 4),
            "range_m": round(range_m, 3)
        })

        frame_idx += 1
        time.sleep(1 / FPS)

    # Final summary
    emit({
        "type": "complete",
        "avg_hr": round(sum(hr_history) / len(hr_history), 1),
        "avg_rr": round(sum(rr_history) / len(rr_history), 1),
        "duration": round(time.time() - start, 1),
        "total_frames": frame_idx
    })

# ────────────────────────────────────────────────────────────
# REAL SENSOR MODE
# ────────────────────────────────────────────────────────────
def run_sensor():
    emit({"type": "status", "message": f"Connecting to radar on {USER_PORT}/{DATA_PORT}..."})

    try:
        user_ser = serial.Serial(USER_PORT, 115200, timeout=2)
        data_ser = serial.Serial(DATA_PORT, 921600, timeout=0)
    except Exception as e:
        emit({"type": "error", "message": f"Serial connection failed: {e}"})
        sys.exit(1)

    time.sleep(1)

    # Stop previous session
    user_ser.write(b"sensorStop\n")
    time.sleep(0.5)

    # Send configuration
    emit({"type": "status", "message": "Sending radar configuration..."})
    if os.path.exists(CFG_FILE):
        with open(CFG_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("%"):
                    user_ser.write((line + "\n").encode())
                    time.sleep(0.03)
    else:
        emit({"type": "error", "message": f"Config file not found: {CFG_FILE}"})
        sys.exit(1)

    user_ser.write(b"sensorStart\n")
    emit({"type": "status", "message": "Radar started. Streaming data..."})

    # Buffers
    rx = bytearray()
    phase_buffer = []  # Rolling raw phase buffer
    hr_history, rr_history = [], []
    range_history = []
    
    # Tracking values
    cur_hr = 0.0
    cur_rr = 0.0
    hw_val = 0.0
    bw_val = 0.0

    start = time.time()
    frame_idx = 0

    while time.time() - start < DURATION:
        rx.extend(data_ser.read(4096))

        while True:
            i = rx.find(MAGIC)
            if i < 0 or len(rx) < i + 40:
                break

            try:
                plen = struct.unpack("<I", rx[i + 12:i + 16])[0]
                if len(rx) < i + plen:
                    break
            except:
                break

            pkt = rx[i:i + plen]
            rx = rx[i + plen:]
            pay = pkt[40:]

            off = 0
            while off + 8 <= len(pay):
                t, l = struct.unpack_from("<II", pay, off)
                off += 8

                if t == 2 and l > 0 and (off + l) <= len(pay):
                    try:
                        # Extract raw I/Q values from Complex Range Profile
                        values = struct.unpack('<' + 'h' * (l // 2), pay[off:off + l])
                        num_bins = l // 4
                        mags = []
                        c_vals = []
                        for b_idx in range(num_bins):
                            iv = values[b_idx * 2]
                            qv = values[b_idx * 2 + 1]
                            c_vals.append((iv, qv))
                            mags.append(np.sqrt(iv*iv + qv*qv))
                        
                        target_bin = int(np.argmax(mags))
                        iv, qv = c_vals[target_bin]
                        phase = float(np.arctan2(qv, iv))
                        
                        range_m = target_bin * 0.044 # Approx distance per bin
                        range_history.append(range_m)
                        range_history = range_history[-10:]
                        range_sm = statistics.median(range_history)

                        phase_buffer.append(phase)
                        phase_buffer = phase_buffer[-BUF_LEN:]

                        # SIGNAL PROCESSING (REDUCED FREQUENCY FOR STABILITY)
                        # Only calculate BPM every 10 frames (2x per second)
                        if len(phase_buffer) >= 40:
                            # 1. Unwrap and Clean
                            clean_phase = detrend(np.unwrap(phase_buffer))
                            
                            # 2. Filter
                            heart_wave = bandpass(clean_phase, 0.8, 2.0, FPS)
                            breath_wave = bandpass(clean_phase, 0.1, 0.5, FPS)
                            
                            hw_val = float(heart_wave[-1])
                            bw_val = float(breath_wave[-1])

                            # 3. BPM Calculation (Warmup check: wait for 5 seconds of data)
                            if frame_idx % 10 == 0 and frame_idx > 100:
                                hr_p, _ = find_peaks(heart_wave, distance=FPS * 0.4)
                                rr_p, _ = find_peaks(breath_wave, distance=FPS * 1.5)
                                
                                dur_window = len(heart_wave) / FPS
                                if len(hr_p) > 1:
                                    cur_hr = (len(hr_p) / dur_window) * 60
                                if len(rr_p) > 1:
                                    cur_rr = (len(rr_p) / dur_window) * 60

                        ts = time.time() - start
                        
                        emit({
                            "type": "frame",
                            "ts": round(ts, 3),
                            "hr": round(cur_hr, 1),
                            "rr": round(cur_rr, 1),
                            "heart_wave": round(hw_val, 4),
                            "breath_wave": round(bw_val, 4),
                            "range_m": round(range_sm, 3)
                        })

                        if frame_idx > 100:
                            hr_history.append(cur_hr)
                            rr_history.append(cur_rr)

                        frame_idx += 1

                    except Exception as e:
                        pass

                off += l

    # Final summary
    if hr_history:
        emit({
            "type": "complete",
            "avg_hr": round(sum(hr_history) / len(hr_history), 1),
            "avg_rr": round(sum(rr_history) / len(rr_history), 1),
            "duration": round(time.time() - start, 1),
            "total_frames": frame_idx
        })

    # Cleanup
    user_ser.write(b"sensorStop\n")
    user_ser.close()
    data_ser.close()

# ────────────────────────────────────────────────────────────
# FILE STREAMING MODE (SYNTHETIC DATA)
# ────────────────────────────────────────────────────────────
def run_file_stream(filepath):
    import csv
    if not os.path.exists(filepath):
        emit({"type": "error", "message": f"File not found: {filepath}"})
        sys.exit(1)
        
    emit({"type": "status", "message": f"Streaming from file: {os.path.basename(filepath)}"})
    
    start = time.time()
    hr_history, rr_history = [], []
    frame_idx = 0
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if time.time() - start >= DURATION:
                break
                
            try:
                hr = float(row.get("HeartRate_BPM", 72))
                rr = float(row.get("RespirationRate_BPM", 16))
                hw = float(row.get("HeartWaveform", 0))
                bw = float(row.get("BreathWaveform", 0))
                range_m = float(row.get("Range_m", 0.7))
                
                hr_history.append(hr)
                rr_history.append(rr)
                
                ts = time.time() - start
                
                emit({
                    "type": "frame",
                    "ts": round(ts, 3),
                    "hr": round(hr, 1),
                    "rr": round(rr, 1),
                    "heart_wave": round(hw, 4),
                    "breath_wave": round(bw, 4),
                    "range_m": round(range_m, 3)
                })
                
                frame_idx += 1
                time.sleep(1 / FPS)
            except ValueError:
                continue

    if hr_history:
        emit({
            "type": "complete",
            "avg_hr": round(sum(hr_history) / len(hr_history), 1),
            "avg_rr": round(sum(rr_history) / len(rr_history), 1),
            "duration": round(time.time() - start, 1),
            "total_frames": frame_idx
        })


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if FILE_MODE:
        run_file_stream(FILE_MODE)
    elif DEMO_MODE:
        run_demo()
    else:
        run_sensor()
