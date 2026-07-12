from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# -------------------------------
# Load YOLO Model Once
# -------------------------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# -------------------------------
# Defect Valuation Matrix
# -------------------------------
DEFECT_MATRIX = {
    "missing cmos battery": {
        "loss": 2.5,
        "fatal": False,
        "label": "Missing CMOS Battery",
        "desc": "Easy CR2032 battery replacement required."
    },
    "loose cpu fan screws": {
        "loss": 1.0,
        "fatal": False,
        "label": "Loose CPU Fan Screws",
        "desc": "Cooling assembly requires tightening."
    },
    "no cpu fan screws": {
        "loss": 3.0,
        "fatal": False,
        "label": "No CPU Fan Screws",
        "desc": "Missing cooler retention hardware."
    },
    "no screw": {
        "loss": 0.5,
        "fatal": False,
        "label": "Missing Screws",
        "desc": "General motherboard screws missing."
    },
    "cpu fan": {
        "loss": 8.0,
        "fatal": False,
        "label": "CPU Fan Defect",
        "desc": "Cooling fan damage or malfunction detected."
    },
    "scratch": {
        "loss": 12.0,
        "fatal": False,
        "label": "PCB Scratch",
        "desc": "PCB trace damage may require repair."
    },
    "missing component": {
        "loss": 18.0,
        "fatal": False,
        "label": "Missing Component",
        "desc": "Electronic component replacement required."
    },
    "rust": {
        "loss": 30.0,
        "fatal": False,
        "label": "Corrosion / Rust",
        "desc": "Chemical cleaning and restoration needed."
    },
    "pin damage": {
        "loss": 60.0,
        "fatal": True,
        "label": "CPU Socket Pin Damage",
        "desc": "Critical CPU socket structural failure."
    },
    "pci damage": {
        "loss": 50.0,
        "fatal": True,
        "label": "PCIe Slot Damage",
        "desc": "Damaged expansion slot causing major failure."
    }
}

DEFAULT_DEFECT = {
    "loss": 10.0,
    "fatal": False,
    "label": "Uncategorized Anomaly",
    "desc": "General hardware anomaly detected."
}


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template(
                "index.html",
                scanned=False,
                error="Please upload a valid image."
            )

        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Invalid image")

            results = model.predict(
                img,
                verbose=False,
                conf=0.25
            )

            detected_defects = []
            img_base64 = ""

            for result in results:

                classes = {
                    model.names[int(cls)]
                    for cls in result.boxes.cls
                }

                classes.discard("motherboard")

                annotated = result.plot()

                success, buffer = cv2.imencode(".png", annotated)

                if success:
                    img_base64 = base64.b64encode(
                        buffer
                    ).decode("utf-8")

                for defect_name in classes:

                    info = DEFECT_MATRIX.get(
                        defect_name.lower(),
                        DEFAULT_DEFECT
                    )

                    detected_defects.append({
                        "raw_name": defect_name,
                        "label": info["label"]
                        if info["label"] != "Uncategorized Anomaly"
                        else defect_name.title(),
                        "loss": info["loss"],
                        "fatal": info["fatal"],
                        "desc": info["desc"]
                    })

            total_loss = sum(
                d["loss"] for d in detected_defects
            )

            has_fatal = any(
                d["fatal"] for d in detected_defects
            )

            if has_fatal or total_loss >= 60:

                status_title = "UNREPAIRABLE (SCRAP ONLY)"

                status_desc = (
                    "Critical structural damage detected. "
                    "Repair costs exceed practical recovery value."
                )

                buy_percentage = 0
                is_repairable = False

            elif not detected_defects:

                total_loss = 0

                status_title = (
                    "PRISTINE / FULLY REPAIRABLE"
                )

                status_desc = (
                    "No hardware anomalies detected. "
                    "Motherboard structure appears healthy."
                )

                buy_percentage = 100
                is_repairable = True

            else:

                status_title = (
                    "REPAIRABLE (DISCOUNT REQUIRED)"
                )

                status_desc = (
                    "Detected defects remain economically "
                    "repairable and suitable for refurbishment."
                )

                buy_percentage = max(
                    100 - total_loss,
                    5
                )

                is_repairable = True

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

        except Exception as e:

            return render_template(
                "index.html",
                scanned=False,
                error=f"Processing failed: {str(e)}"
            )

    return render_template(
        "index.html",
        scanned=False
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )