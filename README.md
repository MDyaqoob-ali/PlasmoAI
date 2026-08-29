# PlasmoAI — Clinical Deep Learning for Malaria Detection 🦟🔬

<p align="center">
  <img src="static/images/icon.png" alt="PlasmoAI Logo" width="120" height="120">
</p>

<p align="center">
  <strong>Production-Grade Medical Artificial Intelligence for Thin Blood Smear Parasite Classification &amp; Cohort Parasitemia Index Screening</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-95.23%25-10B981?style=for-the-badge&logo=tensorflow" alt="Accuracy">
  <img src="https://img.shields.io/badge/Latency-%3C15ms-0EA5E9?style=for-the-badge&logo=fastapi" alt="Latency">
  <img src="https://img.shields.io/badge/Dataset-NIH%20NLM%20(27%2C558%20Slides)-8B5CF6?style=for-the-badge" alt="Dataset">
  <img src="https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge" alt="License">
</p>

---

## 📌 Executive Summary

**PlasmoAI** is a clinical-grade medical AI web application that leverages a **Two-Stage Convolutional Neural Network (CNN)** with Batch Normalization to classify Giemsa-stained thin blood film microscopy images into **Parasitized** (*Plasmodium falciparum*) and **Uninfected** (healthy red blood cells).

The system was trained on **27,558 segmented erythrocyte images** from the official **US National Library of Medicine (NIH) Malaria Screener** research dataset, achieving **95.23% test accuracy**.

---

## ✨ Key Features &amp; Capabilities

- 🔬 **Interactive Diagnostic Laboratory (`/diagnose` &amp; `/form`)**:
  - Drag-and-drop or clipboard paste (`Ctrl+V`) for microscopy slide patches.
  - Built-in Curated Sample Library for 1-click testing of verified Parasitized and Uninfected cells from the NIH holdout set.
  - Real-time image adjustment tools (brightness, contrast, inverted channels, zoom inspection).
  - High-precision calibrated confidence gauges and parasite chromatin hotspot detection maps.
- 🧪 **Batch Smear &amp; Parasitemia Studio (`/batch`)**:
  - Automated high-throughput multi-cell screening.
  - Automated **Parasitemia Index** calculation (`Infected Cells / Total Cells × 100`).
  - Patient risk stratification (Severe, Moderate, Low-density, Clear).
  - One-click CSV and JSON clinical data export.
- 📄 **Clinical Diagnostic Pathology Report Generator (`/result`)**:
  - Printable / Exportable medical-grade PDF pathology reports.
  - Unique report ID generation, specimen dimensions, probability breakdown, clinical recommendations, and doctor signature lines.
- 🧠 **Interactive CNN Architecture Explorer (`/model`)**:
  - Layer-by-layer 16-stage network visualizer with parameter counts and output shapes.
  - Training convergence metrics and confusion matrix data.
- 📚 **Research Background &amp; Dataset Provenance (`/research`)**:
  - Medical rationale, NIH Malaria Screener dataset provenance, and clinical validation protocols.
- ⚡ **High-Performance Native Inference Engine (`malaria_model.py`)**:
  - Sub-15ms inference latency powered by vectorized NumPy forward passes.
  - Zero heavy external runtime dependencies, 100% deterministic mathematical evaluation.

---

## 🏗️ Neural Network Architecture

```
Input (50×50×3 RGB Normalized Tensor)
  │
  ├── Conv2D (1): 32 Filters, 3×3, ReLU ──────────────> (48, 48, 32)
  ├── MaxPooling2D (1): 2×2 Pool, Stride 2 ───────────> (24, 24, 32)
  ├── BatchNormalization (1): Axis -1 ────────────────> (24, 24, 32)
  ├── Dropout (1): Rate 0.2
  │
  ├── Conv2D (2): 32 Filters, 3×3, ReLU ──────────────> (22, 22, 32)
  ├── MaxPooling2D (2): 2×2 Pool, Stride 2 ───────────> (11, 11, 32)
  ├── BatchNormalization (2): Axis -1 ────────────────> (11, 11, 32)
  ├── Dropout (2): Rate 0.2
  │
  ├── Flatten: 11×11×32 Vector ───────────────────────> (3,872)
  │
  ├── Dense (1): 512 Units, ReLU ─────────────────────> (512)
  ├── BatchNormalization (3): Axis 1 ─────────────────> (512)
  ├── Dropout (3): Rate 0.2
  │
  ├── Dense (2): 256 Units, ReLU ─────────────────────> (256)
  ├── BatchNormalization (4): Axis 1 ─────────────────> (256)
  ├── Dropout (4): Rate 0.2
  │
  └── Dense (3) Output: 2 Units (Softmax / Sigmoid) ──> [Parasitized, Uninfected]
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Installation
Clone the repository and install the lightweight requirements:

```bash
git clone https://github.com/HakinCodes/Malaria-Detection.git
cd Malaria-Detection

pip install flask pillow numpy h5py pyfive
```

### 3. Running the Application
Launch the Flask development server:

```bash
python app.py
```

Navigate to **`http://127.0.0.1:5000`** in your browser.

---

## 🧪 Running Automated Tests

Run the complete test suite:

```bash
python run_tests.py
```

---

## 🌐 REST API Reference

### `POST /api/predict`
Diagnose a single blood smear cell image.
- **Request Body (JSON)**: `{"image": "<base64_data_uri_or_sample_filename>"}`
- **Request (Multipart)**: `file: <image_file>`
- **Response**:
```json
{
  "success": true,
  "data": {
    "prediction": "Parasitized",
    "is_parasitized": true,
    "confidence": 99.99,
    "severity": "Critical / High Parasitemia",
    "probabilities": {
      "Parasitized": 99.99,
      "Uninfected": 0.01
    },
    "latency_ms": 12.4
  }
}
```

### `POST /api/predict-batch`
Screen a cohort of blood smear cell images and calculate Parasitemia Index.
- **Request Body (JSON)**: `{"images": ["cell1.png", "cell2.png", ...]}`
- **Request (Multipart)**: `files: [<image1>, <image2>, ...]`
- **Response**:
```json
{
  "success": true,
  "data": {
    "total_cells": 16,
    "infected_count": 8,
    "uninfected_count": 8,
    "parasitemia_index_percent": 50.0,
    "cohort_risk": "Severe Parasitemia (Hospitalization Urgently Indicated)"
  }
}
```

---

## 📜 License &amp; Citations

- **Dataset**: Segmented cell dataset provided by the **US National Library of Medicine (NIH)** and Chittagong Medical College Hospital.
- **License**: [MIT License](LICENSE).
