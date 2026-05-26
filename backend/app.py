from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# =========================
# CLASS UCF
# =========================

classes = [
    "Diving-Side",
    "Golf-Swing-Back",
    "Golf-Swing-Front",
    "Golf-Swing-Side",
    "Kicking-Front",
    "Kicking-Side",
    "Lifting",
    "Riding-Horse",
    "Run-Side",
    "SkateBoarding-Front",
    "Swing-Bench",
    "Swing-SideAngle",
    "Walk-Front"
]

# =========================
# LOAD MODEL
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model_ucf.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=torch.device("cpu")
    )
)

model.eval()

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# API PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "Không có file"
            })

        file = request.files["file"]

        image = Image.open(file).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        action = classes[
            predicted.item()
        ]

        accuracy_value = round(
            confidence.item() * 100,
            2
        )

        accuracy = f"{accuracy_value}%"

        # =========================
        # AI EVALUATION
        # =========================

        if accuracy_value >= 90:

            posture = "Đúng"

            technique = "Tốt"

            evaluation = (
                "Tư thế thực hiện tốt"
            )

            confidence_level = "Rất cao"

        elif accuracy_value >= 70:

            posture = "Khá tốt"

            technique = "Ổn định"

            evaluation = (
                "Động tác tương đối ổn định"
            )

            confidence_level = "Cao"

        else:

            posture = "Chưa ổn định"

            technique = "Cần cải thiện"

            evaluation = (
                "Cần cải thiện động tác"
            )

            confidence_level = "Trung bình"

        # =========================
        # TOP 5
        # =========================

        top_predictions = []

        top_probs, top_classes = torch.topk(
            probabilities,
            5
        )

        for i in range(5):

            top_predictions.append({

                "label":
                classes[
                    top_classes[0][i].item()
                ],

                "score":
                round(
                    top_probs[0][i].item() * 100,
                    2
                )

            })

        return jsonify({

            "prediction": action,

            "accuracy": accuracy,

            "confidence_level":
            confidence_level,

            "posture":
            posture,

            "technique":
            technique,

            "evaluation":
            evaluation,

            "top_predictions":
            top_predictions

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    app.run(debug=True)