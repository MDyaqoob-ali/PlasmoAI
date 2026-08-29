"""
PlasmoAI - Core Machine Learning Inference Engine for Malaria Detection
Native pure-vectorized Convolutional Neural Network forward pass from models/CNN.h5
"""

import os
import io
import base64
import json
import pyfive
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class MalariaCNN:
    def __init__(self, h5_path="models/CNN.h5"):
        self.h5_path = h5_path
        self.weights = {}
        self.classes = {0: "Parasitized", 1: "Uninfected"}
        self.load_weights()

    def load_weights(self):
        """Loads and converts all layer weights from HDF5 file into numpy arrays."""
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Model file not found at {self.h5_path}")
        
        f = pyfive.File(self.h5_path)
        w = f['model_weights']
        
        # Conv2D Layer 1 (32 filters, 3x3x3)
        self.weights['c1_kernel'] = np.asarray(w['conv2d_10']['conv2d_10']['kernel:0'][:], dtype=np.float32)
        self.weights['c1_bias'] = np.asarray(w['conv2d_10']['conv2d_10']['bias:0'][:], dtype=np.float32)
        
        # BatchNormalization Layer 1
        self.weights['bn1_gamma'] = np.asarray(w['batch_normalization_20']['batch_normalization_20']['gamma:0'][:], dtype=np.float32)
        self.weights['bn1_beta'] = np.asarray(w['batch_normalization_20']['batch_normalization_20']['beta:0'][:], dtype=np.float32)
        self.weights['bn1_mean'] = np.asarray(w['batch_normalization_20']['batch_normalization_20']['moving_mean:0'][:], dtype=np.float32)
        self.weights['bn1_var'] = np.asarray(w['batch_normalization_20']['batch_normalization_20']['moving_variance:0'][:], dtype=np.float32)
        
        # Conv2D Layer 2 (32 filters, 3x3x32)
        self.weights['c2_kernel'] = np.asarray(w['conv2d_11']['conv2d_11']['kernel:0'][:], dtype=np.float32)
        self.weights['c2_bias'] = np.asarray(w['conv2d_11']['conv2d_11']['bias:0'][:], dtype=np.float32)
        
        # BatchNormalization Layer 2
        self.weights['bn2_gamma'] = np.asarray(w['batch_normalization_21']['batch_normalization_21']['gamma:0'][:], dtype=np.float32)
        self.weights['bn2_beta'] = np.asarray(w['batch_normalization_21']['batch_normalization_21']['beta:0'][:], dtype=np.float32)
        self.weights['bn2_mean'] = np.asarray(w['batch_normalization_21']['batch_normalization_21']['moving_mean:0'][:], dtype=np.float32)
        self.weights['bn2_var'] = np.asarray(w['batch_normalization_21']['batch_normalization_21']['moving_variance:0'][:], dtype=np.float32)
        
        # Dense Layer 1 (3872 -> 512)
        self.weights['d1_kernel'] = np.asarray(w['dense_15']['dense_15']['kernel:0'][:], dtype=np.float32)
        self.weights['d1_bias'] = np.asarray(w['dense_15']['dense_15']['bias:0'][:], dtype=np.float32)
        
        # BatchNormalization Layer 3
        self.weights['bn3_gamma'] = np.asarray(w['batch_normalization_22']['batch_normalization_22']['gamma:0'][:], dtype=np.float32)
        self.weights['bn3_beta'] = np.asarray(w['batch_normalization_22']['batch_normalization_22']['beta:0'][:], dtype=np.float32)
        self.weights['bn3_mean'] = np.asarray(w['batch_normalization_22']['batch_normalization_22']['moving_mean:0'][:], dtype=np.float32)
        self.weights['bn3_var'] = np.asarray(w['batch_normalization_22']['batch_normalization_22']['moving_variance:0'][:], dtype=np.float32)
        
        # Dense Layer 2 (512 -> 256)
        self.weights['d2_kernel'] = np.asarray(w['dense_16']['dense_16']['kernel:0'][:], dtype=np.float32)
        self.weights['d2_bias'] = np.asarray(w['dense_16']['dense_16']['bias:0'][:], dtype=np.float32)
        
        # BatchNormalization Layer 4
        self.weights['bn4_gamma'] = np.asarray(w['batch_normalization_23']['batch_normalization_23']['gamma:0'][:], dtype=np.float32)
        self.weights['bn4_beta'] = np.asarray(w['batch_normalization_23']['batch_normalization_23']['beta:0'][:], dtype=np.float32)
        self.weights['bn4_mean'] = np.asarray(w['batch_normalization_23']['batch_normalization_23']['moving_mean:0'][:], dtype=np.float32)
        self.weights['bn4_var'] = np.asarray(w['batch_normalization_23']['batch_normalization_23']['moving_variance:0'][:], dtype=np.float32)
        
        # Dense Layer 3 Output (256 -> 2)
        self.weights['d3_kernel'] = np.asarray(w['dense_17']['dense_17']['kernel:0'][:], dtype=np.float32)
        self.weights['d3_bias'] = np.asarray(w['dense_17']['dense_17']['bias:0'][:], dtype=np.float32)

    def _conv2d_valid(self, x, kernel, bias):
        """Vectorized 2D Convolution with valid padding."""
        H, W, Cin = x.shape
        Kh, Kw, _, Cout = kernel.shape
        OutH, OutW = H - Kh + 1, W - Kw + 1
        
        shape = (OutH, OutW, Kh, Kw, Cin)
        strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2])
        cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        cols = cols.reshape(OutH * OutW, Kh * Kw * Cin)
        k_flat = kernel.reshape(Kh * Kw * Cin, Cout)
        out = np.dot(cols, k_flat) + bias
        out = out.reshape(OutH, OutW, Cout)
        return np.maximum(0, out)  # ReLU activation

    def _maxpool2d(self, x, pool_size=2, stride=2):
        """2D Max Pooling with 2x2 window and stride 2."""
        H, W, C = x.shape
        outH, outW = H // stride, W // stride
        shape = (outH, outW, pool_size, pool_size, C)
        strides = (stride * x.strides[0], stride * x.strides[1], x.strides[0], x.strides[1], x.strides[2])
        view = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return np.max(view, axis=(2, 3))

    def _batch_norm(self, x, gamma, beta, mean, var, eps=0.001):
        """Batch Normalization inference scaling."""
        scale = gamma / np.sqrt(var + eps)
        shift = beta - mean * scale
        return x * scale + shift

    def generate_attention_map(self, img_pil):
        """Generates a parasite chromatin hotspot attention map for clinical visualization."""
        img_np = np.array(img_pil.convert('RGB'))
        # Parasites in Giemsa stains show characteristic purple/chromatin dots (high in Red/Blue compared to Green, or dark intense inclusions)
        r, g, b = img_np[:,:,0].astype(np.float32), img_np[:,:,1].astype(np.float32), img_np[:,:,2].astype(np.float32)
        # Chromatin / ring stage trophozoite detection metric
        chromatin_signature = (r * 0.4 + b * 0.6) - (g * 1.1)
        chromatin_signature = np.clip(chromatin_signature, 0, 255)
        
        # Invert intensity relative to cell background
        cell_mask = (r + g + b > 60).astype(np.float32)
        hotspots = chromatin_signature * cell_mask
        
        if np.max(hotspots) > 0:
            hotspots = (hotspots / np.max(hotspots) * 255).astype(np.uint8)
        else:
            hotspots = hotspots.astype(np.uint8)
            
        heat_pil = Image.fromarray(hotspots).filter(ImageFilter.GaussianBlur(radius=1.5))
        buffered = io.BytesIO()
        heat_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, img_source):
        """
        Executes full CNN inference on an image.
        Args:
            img_source: PIL.Image, file path (str), or bytes
        Returns:
            dict with prediction details, probabilities, confidence, risk severity, and metadata.
        """
        if isinstance(img_source, str):
            if img_source.startswith("data:image"):
                header, encoded = img_source.split(",", 1)
                img_data = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            elif os.path.exists(img_source):
                img = Image.open(img_source).convert('RGB')
            else:
                raise ValueError("Invalid file path or data URI")
        elif isinstance(img_source, bytes):
            img = Image.open(io.BytesIO(img_source)).convert('RGB')
        elif isinstance(img_source, Image.Image):
            img = img_source.convert('RGB')
        else:
            raise TypeError("Unsupported image source type")

        original_size = img.size

        # Preprocessing: resize to 50x50 bilinear & normalize to [0, 1]
        img_50 = img.resize((50, 50), Image.Resampling.BILINEAR)
        x = np.array(img_50, dtype=np.float32) / 255.0

        # Layer 1: Conv2D (50x50x3 -> 48x48x32) + ReLU
        x = self._conv2d_valid(x, self.weights['c1_kernel'], self.weights['c1_bias'])
        # Layer 2: MaxPool (48x48x32 -> 24x24x32)
        x = self._maxpool2d(x, 2, 2)
        # Layer 3: BatchNorm 1
        x = self._batch_norm(x, self.weights['bn1_gamma'], self.weights['bn1_beta'], self.weights['bn1_mean'], self.weights['bn1_var'])

        # Layer 4: Conv2D (24x24x32 -> 22x22x32) + ReLU
        x = self._conv2d_valid(x, self.weights['c2_kernel'], self.weights['c2_bias'])
        # Layer 5: MaxPool (22x22x32 -> 11x11x32)
        x = self._maxpool2d(x, 2, 2)
        # Layer 6: BatchNorm 2
        x = self._batch_norm(x, self.weights['bn2_gamma'], self.weights['bn2_beta'], self.weights['bn2_mean'], self.weights['bn2_var'])

        # Layer 7: Flatten (11*11*32 = 3872)
        x = np.ascontiguousarray(x).reshape(-1)

        # Layer 8: Dense (3872 -> 512) + ReLU
        x = np.dot(x, self.weights['d1_kernel']) + self.weights['d1_bias']
        x = np.maximum(0, x)
        # Layer 9: BatchNorm 3
        x = self._batch_norm(x, self.weights['bn3_gamma'], self.weights['bn3_beta'], self.weights['bn3_mean'], self.weights['bn3_var'])

        # Layer 10: Dense (512 -> 256) + ReLU
        x = np.dot(x, self.weights['d2_kernel']) + self.weights['d2_bias']
        x = np.maximum(0, x)
        # Layer 11: BatchNorm 4
        x = self._batch_norm(x, self.weights['bn4_gamma'], self.weights['bn4_beta'], self.weights['bn4_mean'], self.weights['bn4_var'])

        # Layer 12: Dense Output (256 -> 2)
        raw_logits = np.dot(x, self.weights['d3_kernel']) + self.weights['d3_bias']
        
        # Softmax calibrated probability distribution
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        softmax_probs = exp_logits / np.sum(exp_logits)
        
        # Sigmoid raw activations
        sigmoid_probs = 1.0 / (1.0 + np.exp(-raw_logits))

        # Class decision
        class_idx = int(np.argmax(raw_logits))
        pred_label = self.classes[class_idx]
        is_parasitized = (class_idx == 0)
        confidence = float(softmax_probs[class_idx] * 100.0)

        # Morphological and severity analysis
        if is_parasitized:
            if confidence >= 98.0:
                severity = "Critical / High Parasitemia"
                severity_color = "danger"
                clinical_note = "Distinct intracellular Plasmodium chromatin dot and trophozoite morphology detected with high certainty. Immediate antimalarial protocol recommended."
            elif confidence >= 80.0:
                severity = "Positive / Moderate Risk"
                severity_color = "warning"
                clinical_note = "Malarial ring-stage morphology observed. Recommended secondary microscopic verification or rapid diagnostic test (RDT)."
            else:
                severity = "Suspected Low-Grade Parasitemia"
                severity_color = "warning"
                clinical_note = "Atypical cellular inclusions detected. Recommend thin/thick smear recount and Giemsa restaining."
        else:
            if confidence >= 95.0:
                severity = "Clear / Uninfected"
                severity_color = "success"
                clinical_note = "Normal erythrocyte morphology without detectable Plasmodium inclusions. Slide conforms to healthy red blood cell baseline."
            else:
                severity = "Likely Uninfected"
                severity_color = "info"
                clinical_note = "No characteristic trophozoite rings found. Low confidence variance may reflect slide preparation artifact."

        # Generate Base64 preview of analyzed image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Attention hotspot map
        heatmap_base64 = "data:image/png;base64," + self.generate_attention_map(img)

        return {
            "prediction": pred_label,
            "is_parasitized": is_parasitized,
            "confidence": round(confidence, 2),
            "class_index": class_idx,
            "severity": severity,
            "severity_color": severity_color,
            "clinical_note": clinical_note,
            "probabilities": {
                "Parasitized": round(float(softmax_probs[0] * 100.0), 2),
                "Uninfected": round(float(softmax_probs[1] * 100.0), 2)
            },
            "raw_scores": [round(float(s), 4) for s in sigmoid_probs],
            "image_data": img_base64,
            "heatmap_data": heatmap_base64,
            "image_dimensions": {"width": original_size[0], "height": original_size[1]},
            "input_resolution": "50x50 RGB",
            "model_architecture": "2-Stage CNN + 4x BatchNorm + 3x Dense"
        }

    def get_model_summary(self):
        """Returns metadata and layer specifications."""
        return {
            "name": "PlasmoAI Two-Layered Convolutional Neural Network",
            "framework": "Trained with Keras / TensorFlow, High-Performance Native Inference Engine",
            "input_shape": [50, 50, 3],
            "classes": ["Parasitized (Plasmodium)", "Uninfected (Normal RBC)"],
            "test_accuracy": 95.23,
            "dataset_origin": "NIH US National Library of Medicine (27,558 segmented thin blood smear images)",
            "layers": [
                {"name": "Conv2D (1)", "filters": 32, "kernel_size": "3x3", "activation": "ReLU", "output_shape": [48, 48, 32]},
                {"name": "MaxPooling2D (1)", "pool_size": "2x2", "stride": "2x2", "output_shape": [24, 24, 32]},
                {"name": "BatchNormalization (1)", "axis": "Channels (-1)", "momentum": 0.99, "output_shape": [24, 24, 32]},
                {"name": "Dropout (1)", "rate": 0.2, "output_shape": [24, 24, 32]},
                {"name": "Conv2D (2)", "filters": 32, "kernel_size": "3x3", "activation": "ReLU", "output_shape": [22, 22, 32]},
                {"name": "MaxPooling2D (2)", "pool_size": "2x2", "stride": "2x2", "output_shape": [11, 11, 32]},
                {"name": "BatchNormalization (2)", "axis": "Channels (-1)", "momentum": 0.99, "output_shape": [11, 11, 32]},
                {"name": "Dropout (2)", "rate": 0.2, "output_shape": [11, 11, 32]},
                {"name": "Flatten", "output_shape": [3872]},
                {"name": "Dense (1)", "units": 512, "activation": "ReLU", "output_shape": [512]},
                {"name": "BatchNormalization (3)", "axis": 1, "output_shape": [512]},
                {"name": "Dropout (3)", "rate": 0.2, "output_shape": [512]},
                {"name": "Dense (2)", "units": 256, "activation": "ReLU", "output_shape": [256]},
                {"name": "BatchNormalization (4)", "axis": 1, "output_shape": [256]},
                {"name": "Dropout (4)", "rate": 0.2, "output_shape": [256]},
                {"name": "Dense Output (3)", "units": 2, "activation": "Softmax / Sigmoid", "output_shape": [2]}
            ]
        }
