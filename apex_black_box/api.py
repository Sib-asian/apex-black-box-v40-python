"""
Flask API – exposes the Oracle Engine as a local HTTP endpoint.

POST /api/scan
  Body: JSON payload (see apex_black_box/engine.py for field docs)
  Returns: JSON result from engine.scan()

GET /api/health
  Returns: {"status": "ok"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from apex_black_box import engine

app = Flask(__name__)
CORS(app)  # allow same-machine requests from the Streamlit iframe


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/scan", methods=["POST"])
def scan():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = engine.scan(payload)
        return jsonify(result)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500
