#replaced app.py
from flask import Flask, request, jsonify
import pickle
import time

from prometheus_client import Counter, Histogram, generate_latest


app = Flask(__name__)


# Load trained model
with open("model/iris_model.pkl", "rb") as file:
    model = pickle.load(file)


# Iris class names
class_names = [
    "setosa",
    "versicolor",
    "virginica"
]


# -----------------------------
# Prometheus Metrics
# -----------------------------

# Total number of prediction requests
prediction_requests = Counter(
    "iris_prediction_requests_total",
    "Total number of Iris prediction requests"
)


# Total number of prediction errors
prediction_errors = Counter(
    "iris_prediction_errors_total",
    "Total number of Iris prediction errors"
)


# Prediction request latency
prediction_latency = Histogram(
    "iris_prediction_latency_seconds",
    "Time taken to generate an Iris prediction"
)


# Number of predictions for each class
prediction_class = Counter(
    "iris_predictions_by_class_total",
    "Total predictions by Iris class",
    ["class_name"]
)


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return "Iris ML Monitoring API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    prediction_requests.inc()

    start_time = time.time()

    try:
        data = request.get_json()

        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]

        prediction = model.predict([features])[0]

        predicted_class = class_names[prediction]

        prediction_class.labels(
            class_name=predicted_class
        ).inc()

        return jsonify({
            "prediction": int(prediction),
            "class": predicted_class
        })

    except Exception as e:

        prediction_errors.inc()

        return jsonify({
            "error": str(e)
        }), 400

    finally:

        prediction_latency.observe(
            time.time() - start_time
        )


# Prometheus metrics endpoint
@app.route("/metrics")
def metrics():

    return generate_latest(), 200, {
        "Content-Type": "text/plain"
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )