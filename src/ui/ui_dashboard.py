from flask import Flask, render_template, jsonify
import config
from db import get_latest_status, get_status_history
from config import logger

app = Flask(__name__)

@app.route("/")
def dashboard():
    try:
        status = get_latest_status() or {'detections': 0, 'cpu_usage': 0, 'uptime': 'N/A'}
        return render_template("dashboard.html", status=status)
    except Exception as e:
        logger.error("Dashboard rendering failed: " + str(e))
        return "Error rendering dashboard", 500

@app.route("/api/status/latest")
def api_latest_status():
    try:
        status = get_latest_status()
        if status is None:
            return jsonify({'error': 'No data available'}), 404
        return jsonify(status)
    except Exception as e:
        logger.error("API latest status error: " + str(e))
        return jsonify({'error': 'Error fetching latest status'}), 500

@app.route("/api/status/history")
def api_status_history():
    try:
        history = get_status_history()
        return jsonify(history)
    except Exception as e:
        logger.error("API status history error: " + str(e))
        return jsonify({'error': 'Error fetching status history'}), 500
from iads.db import get_latest_status, get_status_history
from iads.config import logger

app = Flask(__name__)

@app.route("/")
def dashboard():
    try:
        # Fetch system statistics; in a real implementation,
        # gather metrics from a database or log files.
        system_status = {
            "detections": 5,
            "uptime": "2 hours 15 minutes",
            "cpu_usage": "45%"
        }
        return render_template("dashboard.html", status=system_status)
    except Exception as e:
        logger.error("Dashboard rendering failed: " + str(e))
        return "Error rendering dashboard", 500

@app.route("/logs")
def get_logs():
    try:
        # In a real system, read logs from a file or logging service.
        logs = ["Log entry 1", "Log entry 2", "Log entry 3"]
        return jsonify(logs)
    except Exception as e:
        logger.error("Failed to fetch logs: " + str(e))
        return jsonify({"error": "Log retrieval failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
