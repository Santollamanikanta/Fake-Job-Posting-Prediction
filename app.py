import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

# MUST be before tensorflow import for some environments
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# ===============================
# LOAD TOKENIZER
# ===============================
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")
model = tf.keras.models.load_model("fake_job_lstm_model.h5")
print("Model loaded successfully!")

# ===============================
# CONSTANTS
# ===============================
MAX_SEQUENCE_LENGTH = 200

# ===============================
# PREPROCESSING
# ===============================
def preprocess_text(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )
    return padded

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET"])
def home():
    print("Home route accessed")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    combined_text = request.form.get("combined_text")
    print(f"Prediction requested for: {combined_text[:50]}..." if combined_text else "Empty input")

    if not combined_text or combined_text.strip() == "":
        return render_template(
            "index.html",
            prediction="❗ Please enter the job description."
        )

    input_data = preprocess_text(combined_text)

    prediction = model.predict(input_data, verbose=0)[0][0]

    result = "Fraudulent 🚨" if prediction > 0.7 else "Legitimate ✅"
    print(f"Prediction result: {result}")

    return render_template(
        "index.html",
        prediction=f"The job post is: {result}"
    )

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
