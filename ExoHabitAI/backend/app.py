from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
from utils import build_input_dataframe, validate_input

app = Flask(__name__)
CORS(app)

# Load model and feature columns
model = joblib.load("models/habitability_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return "ExoHabitAI Backend running (structured)"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    is_valid, error_message = validate_input(data)
    if not is_valid:
        return jsonify({
            "status": "error",
            "message": error_message
        }), 400

    input_df = build_input_dataframe(data, feature_columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    # ✅ probability support confirmed
    confidence = float(model.predict_proba(input_scaled)[0][1])

    return jsonify({
        "prediction": int(prediction),
        "confidence": round(confidence * 100, 2)
    })


@app.route("/rank", methods=["POST"])
def rank_planets():
    data = request.get_json()

    if "planets" not in data or not isinstance(data["planets"], list):
        return jsonify({
            "status": "error",
            "message": "Input must contain a list of planets"
        }), 400

    ranked = []

    for planet in data["planets"]:
        is_valid, error_message = validate_input(planet)
        if not is_valid:
            continue  # skip invalid planet

        input_df = build_input_dataframe(planet, feature_columns)
        input_scaled = scaler.transform(input_df)

        # probability-based score (if supported)
        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(input_scaled)[0][1])
        else:
            score = float(model.predict(input_scaled)[0])

        ranked.append({
            "name": planet.get("name", "unknown"),
            "score": round(score, 3)
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "status": "success",
        "ranked_planets": ranked
    })

if __name__ == "__main__":
    app.run(debug=True)

