from flask import Flask, render_template, request
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Load your local YOLOv8 model weights
model = YOLO("best.pt")

# --- DEFECT VALUATION MATRIX ---
# Maps exact model classes to severity percentage discounts and repairability limits.
DEFECT_MATRIX = {
    "missing cmos battery": {"loss": 2.5, "fatal": False, "label": "Missing CMOS Battery", "desc": "Easy CR2032 coin cell replacement"},
    "loose cpu fan screws": {"loss": 1.0, "fatal": False, "label": "Loose CPU Fan Screws", "desc": "Minor mounting hardware adjustment needed"},
    "no cpu fan screws": {"loss": 3.0, "fatal": False, "label": "No CPU Fan Screws", "desc": "Missing cooler retention hardware"},
    "no screw": {"loss": 0.5, "fatal": False, "label": "Missing Screws", "desc": "Missing general standoffs or mounting screws"},
    "cpu fan": {"loss": 8.0, "fatal": False, "label": "CPU Fan Defect", "desc": "Faulty or damaged cooling fan mechanism"},
    "scratch": {"loss": 12.0, "fatal": False, "label": "PCB Scratch", "desc": "Surface trace scratch (requires solder mask restoration)"},
    "missing component": {"loss": 18.0, "fatal": False, "label": "Missing Component", "desc": "Requires micro-soldering replacement component"},
    "rust": {"loss": 30.0, "fatal": False, "label": "Oxidation / Rust", "desc": "Corrosion present; requires chemical ultrasonic bath"},
    "pin damage": {"loss": 60.0, "fatal": True, "label": "CPU Socket Pin Damage", "desc": "Bent or broken CPU socket pins (Critical failure risk)"},
    "pci damage": {"loss": 50.0, "fatal": True, "label": "PCIe Slot Damage", "desc": "Cracked or ripped PCIe slot (Structural failure)"}
}

DEFAULT_DEFECT = {"loss": 10.0, "fatal": False, "label": "Uncategorized Anomaly", "desc": "General physical anomaly detected"}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == '':
            return render_template("index.html", error="Please upload a valid image.")

        # Read image stream directly into OpenCV format in memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLOv8 inference
        results = model.predict(img, verbose=False)
        
        detected_defects = []
        img_base64 = ""

        for r in results:
            # Extract unique detected classes (ignoring general 'motherboard' tag)
            raw_classes = [model.names[int(cls)] for cls in r.boxes.cls]
            unique_classes = list(set([d for d in raw_classes if d.lower() != "motherboard"]))
            
            # Generate bounding box image
            annotated_img = r.plot()
            _, buffer = cv2.imencode('.png', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Build diagnostic list with parallel percentage loss values
            for d_name in unique_classes:
                key = d_name.lower()
                info = DEFECT_MATRIX.get(key, DEFAULT_DEFECT.copy())
                if info["label"] == "Uncategorized Anomaly":
                    info["label"] = d_name.title()
                detected_defects.append({
                    "raw_name": d_name,
                    "label": info["label"],
                    "loss": info["loss"],
                    "fatal": info["fatal"],
                    "desc": info["desc"]
                })

        # --- REPAIRABILITY & PERCENTAGE CALCULATIONS ---
        total_loss = sum(d["loss"] for d in detected_defects)
        has_fatal = any(d["fatal"] for d in detected_defects)
        
        # Unrepairable if a fatal structural defect exists OR if total discount exceeds 60%
        if has_fatal or total_loss >= 60.0:
            is_repairable = False
            status_title = "UNREPAIRABLE (SCRAP ONLY)"
            status_desc = "Critical structural damage or excessive restoration labor required. Recommend routing to raw metal recycling."
            buy_percentage = 0.0
            total_loss = min(total_loss, 100.0)
        elif len(detected_defects) == 0:
            is_repairable = True
            status_title = "PRISTINE / FULLY REPAIRABLE"
            status_desc = "No surface defects detected. Hardware architecture is structurally sound."
            total_loss = 0.0
            buy_percentage = 100.0
        else:
            is_repairable = True
            status_title = "REPAIRABLE (DISCOUNT REQUIRED)"
            status_desc = "Salvageable surface defects identified. Economically viable for refurbishment."
            buy_percentage = max(100.0 - total_loss, 5.0)

        return render_template(
            "index.html",
            scanned=True,
            defects=detected_defects,
            img_data=img_base64,
            is_repairable=is_repairable,
            status_title=status_title,
            status_desc=status_desc,
            total_loss=round(total_loss, 1),
            buy_percentage=round(buy_percentage, 1)
        )

    return render_template("index.html", scanned=False)

if __name__ == "__main__":
    app.run(debug=True)