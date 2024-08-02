import io
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import ObjectPrediction, get_sliced_prediction


class Model:
    def __init__(self) -> None:
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            # Download the pytorch file from ultralytics hub and put it
            # in the same directory level as main.py
            model_path="scratch100.pt",
            confidence_threshold=0.2,
            # Adjust the device accordingly to where this backend server
            # is deployed and whether you want to use CPU/GPU
            device="cpu",
        )

    def convertData(self, data):
        return data.astype(np.float64) if isinstance(data, np.float32) else data

    def convertToResponse(self, result: ObjectPrediction, imageWidth, imageHeight):
        return {
            "bbox": [
                self.convertData(result.bbox.minx),
                self.convertData(result.bbox.miny),
                self.convertData(result.bbox.maxx),
                self.convertData(result.bbox.maxy),
            ],
            "category": result.category.name,
            "score": result.score.value,
            "imageWidth": imageWidth,
            "imageHeight": imageHeight,
        }

    def detect(self, image: Image, image_np):
        results = get_sliced_prediction(
            image_np,
            model.detection_model,
            slice_height=image.height // 2,
            slice_width=image.width // 2,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        transformedRes = list(
            map(
                lambda result: self.convertToResponse(
                    result, image.width, image.height
                ),
                results.object_prediction_list,
            )
        )
        return transformedRes


# Initialize flask server and routes.
app = Flask(__name__)
model = Model()
CORS(app)


@app.route("/detect", methods=["POST"])
def detect():
    allDetections = []
    # List of XFiles sent from flutter.
    files = request.files.getlist("images")
    print(f"============================= Making prediction for {len(files)} images =============================")
    for i, file in enumerate(files):
        image = Image.open(io.BytesIO(file.read()))
        # List of detections for each XFile.
        result = model.detect(image, np.array(image))
        allDetections.append(result)
        print(f"=================================== Image {i + 1} Prediction ===================================")
        print(result)
    print("===================================== All Detections =====================================")
    print(allDetections)
    return jsonify(allDetections), 200


@app.route("/", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
