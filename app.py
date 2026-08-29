"""
PlasmoAI - Clinical Deep Learning Web Application for Malaria Detection
Flask Application Entry Point and API Router
"""

import os
import json
import time
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from malaria_model import MalariaCNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max batch upload

# Load Project Configuration
try:
    with open("config.json", "r") as c:
        config_data = json.load(c)
        project_info = config_data.get("project", {})
except Exception:
    project_info = {"name": "PlasmoAI", "version": "2.0.0"}

# Load Curated Samples Catalog
try:
    with open("curated_samples.json", "r") as f:
        curated_samples = json.load(f)
except Exception:
    curated_samples = {"parasitized": [], "uninfected": []}

# Initialize CNN Inference Engine
cnn_engine = MalariaCNN("models/CNN.h5")


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route("/")
def landing_page():
    """Modern clinical landing page with hero scanner and interactive showcase."""
    return render_template(
        "index.html",
        title="PlasmoAI — Deep Learning Malaria Detection Platform",
        curated_samples=curated_samples,
        project_info=project_info
    )


@app.route("/form")
@app.route("/diagnose")
def diagnostic_lab():
    """Interactive diagnostic workbench for single cell and batch microscopy analysis."""
    sample_file = request.args.get("sample")
    return render_template(
        "form.html",
        title="Diagnostic Laboratory — PlasmoAI",
        curated_samples=curated_samples,
        preselected_sample=sample_file,
        project_info=project_info
    )


@app.route("/result")
def result_page():
    """Detailed diagnostic report view."""
    return render_template(
        "result.html",
        title="Diagnostic Report — PlasmoAI",
        curated_samples=curated_samples,
        project_info=project_info
    )


@app.route("/batch")
def batch_analysis():
    """Batch smear and parasitemia cohort analysis studio."""
    return render_template(
        "batch.html",
        title="Batch Smear & Parasitemia Studio — PlasmoAI",
        curated_samples=curated_samples,
        project_info=project_info
    )


@app.route("/model")
def model_explorer():
    """Interactive CNN Architecture and training evaluation explorer."""
    model_summary = cnn_engine.get_model_summary()
    return render_template(
        "model_explorer.html",
        title="CNN Model Architecture — PlasmoAI",
        model_summary=model_summary,
        project_info=project_info
    )


@app.route("/research")
@app.route("/team")
def research_page():
    """Clinical research methodology, dataset provenance, and validation metrics."""
    return render_template(
        "research.html",
        title="Research & Dataset — PlasmoAI",
        project_info=project_info
    )


@app.route("/holdout_images/<path:filename>")
def serve_holdout_image(filename):
    """Serve microscopic cell images from the holdout dataset folder."""
    return send_from_directory("holdout_dataset", filename)


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Classify a single microscopic blood cell image.
    Accepts multipart form-data 'file' or JSON payload with 'image' (base64 string or sample filename).
    """
    start_time = time.perf_counter()
    
    try:
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            img_bytes = file.read()
            result = cnn_engine.predict(img_bytes)
            filename = file.filename
        elif request.is_json:
            data = request.get_json()
            if not data or "image" not in data:
                return jsonify({"error": "Missing 'image' in JSON payload"}), 400
            
            image_val = data["image"]
            if isinstance(image_val, str) and not image_val.startswith("data:image"):
                sample_path = os.path.join("holdout_dataset", os.path.basename(image_val))
                if os.path.exists(sample_path):
                    result = cnn_engine.predict(sample_path)
                    filename = os.path.basename(image_val)
                else:
                    return jsonify({"error": f"Sample image '{image_val}' not found"}), 404
            else:
                result = cnn_engine.predict(image_val)
                filename = data.get("filename", "uploaded_cell.png")
        else:
            return jsonify({"error": "Unsupported request format. Send multipart form or JSON"}), 400

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result["latency_ms"] = elapsed_ms
        result["filename"] = filename
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    """
    Classify a batch of cell images and compute aggregated Parasitemia Index.
    Accepts multiple files in multipart 'files' or JSON list of sample names / base64 strings.
    """
    start_time = time.perf_counter()
    results = []
    infected_count = 0
    uninfected_count = 0

    try:
        if "files" in request.files:
            files = request.files.getlist("files")
            if not files:
                return jsonify({"error": "No files uploaded"}), 400

            for file in files:
                if file.filename:
                    img_bytes = file.read()
                    res = cnn_engine.predict(img_bytes)
                    res["filename"] = file.filename
                    if res["is_parasitized"]:
                        infected_count += 1
                    else:
                        uninfected_count += 1
                    results.append({
                        "filename": file.filename,
                        "prediction": res["prediction"],
                        "confidence": res["confidence"],
                        "is_parasitized": res["is_parasitized"],
                        "severity": res["severity"],
                        "probabilities": res["probabilities"],
                        "image_data": res["image_data"]
                    })
        elif request.is_json:
            data = request.get_json()
            images = data.get("images", [])
            if not images:
                return jsonify({"error": "Empty images list"}), 400

            for item in images:
                if isinstance(item, str) and not item.startswith("data:image"):
                    sample_path = os.path.join("holdout_dataset", os.path.basename(item))
                    if os.path.exists(sample_path):
                        res = cnn_engine.predict(sample_path)
                        fn = os.path.basename(item)
                    else:
                        continue
                else:
                    res = cnn_engine.predict(item)
                    fn = "batch_cell.png"

                if res["is_parasitized"]:
                    infected_count += 1
                else:
                    uninfected_count += 1

                results.append({
                    "filename": fn,
                    "prediction": res["prediction"],
                    "confidence": res["confidence"],
                    "is_parasitized": res["is_parasitized"],
                    "severity": res["severity"],
                    "probabilities": res["probabilities"],
                    "image_data": res["image_data"]
                })
        else:
            return jsonify({"error": "Invalid payload format"}), 400

        total_cells = len(results)
        parasitemia_rate = round((infected_count / total_cells * 100), 2) if total_cells > 0 else 0.0
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if parasitemia_rate >= 20.0:
            cohort_risk = "Severe Parasitemia (Hospitalization Urgently Indicated)"
            cohort_risk_color = "danger"
        elif parasitemia_rate > 2.0:
            cohort_risk = "Moderate Parasitemia (Uncomplicated Malaria Treatment)"
            cohort_risk_color = "warning"
        elif parasitemia_rate > 0.0:
            cohort_risk = "Low-Density Parasitemia (Early / Submicroscopic Infection)"
            cohort_risk_color = "warning"
        else:
            cohort_risk = "Non-Infected Cohort (0% Parasitemia)"
            cohort_risk_color = "success"

        return jsonify({
            "success": True,
            "data": {
                "total_cells": total_cells,
                "infected_count": infected_count,
                "uninfected_count": uninfected_count,
                "parasitemia_index_percent": parasitemia_rate,
                "cohort_risk": cohort_risk,
                "cohort_risk_color": cohort_risk_color,
                "execution_time_ms": elapsed_ms,
                "results": results
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sample-images", methods=["GET"])
def api_sample_images():
    """Returns curated gallery of holdout dataset microscopy images."""
    return jsonify({
        "success": True,
        "data": curated_samples
    })


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Returns model architecture, parameters, and training metrics."""
    return jsonify({
        "success": True,
        "data": cnn_engine.get_model_summary()
    })


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(e):
    return render_template(
        "404.html",
        title="404 Page Not Found — PlasmoAI",
        project_info=project_info
    ), 404


@app.errorhandler(500)
def server_error(e):
    return render_template(
        "404.html",
        title="500 Server Error — PlasmoAI",
        error_message="An internal diagnostic engine error occurred. Please refresh or check server logs.",
        project_info=project_info
    ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
