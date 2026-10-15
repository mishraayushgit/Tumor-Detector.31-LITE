import os
import time
import uuid
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
# Enable CORS for all domains so Vercel frontend can call it smoothly
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "/tmp/uploads" if os.name != "nt" else "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Exact ResNet50 dataset class order
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Load Lightweight TFLite Model (Only 23MB · Ultra Fast) ───────────────────
print("Initializing Tumor Detector Lite AI Engine...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tumor_detector.tflite")

try:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[OK] TFLite Model loaded successfully from {MODEL_PATH} ({size_mb:.1f} MB)!")
except Exception as e:
    print(f"[ERROR] Error loading TFLite model: {e}")
    interpreter = None


# ── Prediction Function ───────────────────────────────────────────────────────
def predict_tumor(image_path):
    t0 = time.time()

    # Preprocess with Pillow directly (fast & memory efficient)
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((224, 224))
        arr = np.array(im, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

    # Fast TFLite inference
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(probs))
    label = class_labels[predicted_index]
    confidence = float(probs[predicted_index])
    result = "No Tumor" if label == "notumor" else f"Tumor: {label}"
    probabilities = {cls: float(probs[i]) for i, cls in enumerate(class_labels)}

    elapsed = time.time() - t0
    print(f"Prediction: {result} ({confidence*100:.2f}%) in {elapsed:.3f}s")
    return result, label, confidence, probabilities


# ── Detection Logic Handler ───────────────────────────────────────────────────
def handle_detection():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    safe_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(filepath)

    try:
        result, label, confidence, probabilities = predict_tumor(filepath)
        return jsonify({
            "result": result,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities
        })
    except Exception as err:
        print("Inference error:", err)
        return jsonify({"error": str(err)}), 500


# ── Routes ────────────────────────────────────────────────────────────────────
# Root route supports both GET (health check) and POST (fallback detection)
@app.route("/", methods=["GET", "POST"])
def health_or_detect():
    if request.method == "POST":
        return handle_detection()

    return jsonify({
        "status": "online",
        "service": "Tumor Detector Lite AI Backend",
        "model": "ResNet50 TFLite (23.3 MB)",
        "memory_optimized": True,
        "endpoints": {
            "detect": "POST /api/detect (form-data: 'file')"
        }
    })


# Primary detection endpoint
@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect_api():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    return handle_detection()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
