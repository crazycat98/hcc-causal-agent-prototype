"""Small local HTTP service for M1-M3.

Safety: 演示用合成数据，非真实患者数据；非临床诊断依据。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .explanation import explain_one
from .features import SYNTHETIC_DATA_NOTICE
from .predictor import ModelBundle, load_model, predict_one


class PredictionHandler(BaseHTTPRequestHandler):
    bundle: ModelBundle

    def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "safety_notice": SYNTHETIC_DATA_NOTICE,
                    "features_required": self.bundle.selected_features,
                    "endpoints": ["/predict", "/explain"],
                },
            )
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in ("/predict", "/explain"):
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(raw_body)
            features = body.get("features")
            if not isinstance(features, dict):
                raise ValueError("Request body must include a features object.")

            if self.path == "/predict":
                response = predict_one(self.bundle, features)
            else:
                response = explain_one(self.bundle, features, top_n=int(body.get("top_n", 5)))

            response["patient_id"] = body.get("patient_id")
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body.", "safety_notice": SYNTHETIC_DATA_NOTICE})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc), "safety_notice": SYNTHETIC_DATA_NOTICE})
        except Exception as exc:  # pragma: no cover - defensive service boundary.
            self._send_json(500, {"error": f"ML service failed: {exc}", "safety_notice": SYNTHETIC_DATA_NOTICE})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(artifact_path: Path, host: str = "127.0.0.1", port: int = 8001) -> None:
    PredictionHandler.bundle = load_model(artifact_path)
    server = ThreadingHTTPServer((host, port), PredictionHandler)
    print(f"M1-M3 ML service running at http://{host}:{port}")
    print(SYNTHETIC_DATA_NOTICE)
    server.serve_forever()

