"""
realtime_server.py — Flask-SocketIO server for real-time vital signs streaming.

Spawns sensor_stream.py as a subprocess, reads JSON lines from its stdout,
and emits them to the browser via WebSocket events.

Usage:
  python realtime_server.py          # Starts on port 5050
  python realtime_server.py --demo   # Forces sensor into demo mode
"""

import os
import sys
import json
import subprocess
import threading
import argparse
import joblib
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import werkzeug.utils

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SENSOR_SCRIPT = os.path.join(WORK_DIR, "sensor_stream.py")
FRONTEND_DIR = os.path.join(WORK_DIR, "frontend")
UPLOAD_DIR = os.path.join(WORK_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

REGRESSOR_PATH = os.path.join(WORK_DIR, "position_aware_hr_model.joblib")
CLASSIFIER_PATH = os.path.join(WORK_DIR, "detectability_classifier.joblib")
DEFAULT_CFG = os.path.join(WORK_DIR, "config", "xwr68xx_profile_VitalSigns_20fps_Front.cfg")

parser = argparse.ArgumentParser()
parser.add_argument("--demo", action="store_true", help="Force demo mode")
cli_args = parser.parse_args()

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Load ML models
regressor = joblib.load(REGRESSOR_PATH) if os.path.exists(REGRESSOR_PATH) else None
classifier = joblib.load(CLASSIFIER_PATH) if os.path.exists(CLASSIFIER_PATH) else None

sensor_process = None

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    from flask import request, jsonify
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = werkzeug.utils.secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        return jsonify({"success": True, "filepath": filepath, "filename": filename})
    
    return jsonify({"error": "Invalid file format, must be CSV"}), 400

# ────────────────────────────────────────────────────────────
# WEBSOCKET EVENTS
# ────────────────────────────────────────────────────────────
@socketio.on("start_sensor")
def handle_start(data):
    global sensor_process

    if sensor_process and sensor_process.poll() is None:
        emit("status", {"message": "Sensor is already running!"})
        return

    demo_flag = cli_args.demo or data.get("demo", False)
    duration = data.get("duration", 30)
    filepath = data.get("filepath")

    cmd = [sys.executable, SENSOR_SCRIPT, "--duration", str(duration)]
    
    if filepath and os.path.exists(filepath):
        cmd.extend(["--file", filepath])
        emit("status", {"message": f"Streaming from uploaded file..."})
    elif demo_flag:
        cmd.append("--demo")
        emit("status", {"message": f"Starting sensor (demo mode)..."})
    else:
        cmd.extend(["--cfg", DEFAULT_CFG])
        emit("status", {"message": f"Starting sensor (live mode)..."})

    sensor_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Stream output in a background thread
    def stream_output():
        try:
            for line in sensor_process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msg_type = obj.get("type", "unknown")

                    if msg_type == "frame":
                        socketio.emit("vital_frame", obj)
                    elif msg_type == "complete":
                        # Enrich with ML predictions if models are loaded
                        ml_results = run_ml_predictions(obj, data)
                        obj["ml_results"] = ml_results
                        socketio.emit("session_complete", obj)
                    elif msg_type == "status":
                        socketio.emit("status", {"message": obj.get("message", "")})
                    elif msg_type == "error":
                        socketio.emit("sensor_error", {"message": obj.get("message", "")})
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            socketio.emit("sensor_error", {"message": str(e)})

    thread = threading.Thread(target=stream_output, daemon=True)
    thread.start()

@socketio.on("stop_sensor")
def handle_stop():
    global sensor_process
    if sensor_process and sensor_process.poll() is None:
        sensor_process.terminate()
        emit("status", {"message": "Sensor stopped by user."})

# ────────────────────────────────────────────────────────────
# ML PREDICTIONS
# ────────────────────────────────────────────────────────────
def run_ml_predictions(complete_data, start_data):
    """Run ML models on the session averages."""
    results = {}
    import pandas as pd

    posture = start_data.get("posture", "Lying_Supine")
    sensor_pos = start_data.get("sensorPosition", "Chest")
    orientation = start_data.get("orientation", 0)
    avg_hr = complete_data.get("avg_hr", 72)

    if regressor:
        try:
            df = pd.DataFrame([{
                "HeartRate_BPM": avg_hr,
                "Posture": posture,
                "SensorPosition": sensor_pos,
                "Orientation_deg": orientation,
                "Distance_m": 0.7,
                "SNR_dB": -30,
                "WaveEnergy": 0.001
            }])
            refined_hr = float(regressor.predict(df)[0])
            results["refined_hr"] = round(refined_hr, 1)
            results["hr_improvement"] = round(abs(avg_hr - refined_hr), 1)
        except Exception as e:
            results["regressor_error"] = str(e)

    if classifier:
        try:
            df = pd.DataFrame([{
                "Posture": posture,
                "SensorPosition": sensor_pos,
                "Orientation_deg": orientation,
                "Distance_m": 0.7,
                "SNR_dB": -30,
                "WaveEnergy": 0.001
            }])
            prob = float(classifier.predict_proba(df)[0][1])
            results["detectability"] = round(prob * 100, 1)
        except Exception as e:
            results["classifier_error"] = str(e)

    return results

# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Radarix Real-Time Server")
    print(f"  Models: Regressor={'✅' if regressor else '❌'}  Classifier={'✅' if classifier else '❌'}")
    print(f"  Mode: {'DEMO' if cli_args.demo else 'LIVE'}")
    print(f"  Dashboard: http://localhost:5050")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)
