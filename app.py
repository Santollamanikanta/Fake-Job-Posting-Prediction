import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# ===============================
# LOAD ARTIFACTS
# ===============================
print("Loading artifacts...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load NumPy weights
weights = np.load("model_weights.npz")
print("Artifacts loaded successfully!")

# ===============================
# CONSTANTS
# ===============================
MAX_SEQUENCE_LENGTH = 200

# ===============================
# NUMPY INFERENCE ENGINE
# ===============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def predict_numpy(text):
    # Preprocessing
    seq = tokenizer.texts_to_sequences([text])[0]
    # Manual padding (post)
    if len(seq) > MAX_SEQUENCE_LENGTH:
        seq = seq[:MAX_SEQUENCE_LENGTH]
    else:
        seq = seq + [0] * (MAX_SEQUENCE_LENGTH - len(seq))
    
    # 1. Embedding Layer
    emb_weights = weights['embedding_weights']
    x = emb_weights[seq] # (200, 100)
    
    # 2. LSTM Layer
    kernel = weights['lstm_kernel']
    rec_kernel = weights['lstm_recurrent_kernel']
    bias = weights['lstm_bias']
    units = rec_kernel.shape[0]
    
    h = np.zeros(units)
    c = np.zeros(units)
    
    # LSTM loop
    for x_t in x:
        # z = x*W + h*U + b
        z = np.dot(x_t, kernel) + np.dot(h, rec_kernel) + bias
        # Split gates: Keras order is i, f, c, o
        # (Actually Keras LSTM uses i, f, c, o by default)
        i = sigmoid(z[:units])
        f = sigmoid(z[units:2*units])
        g = np.tanh(z[2*units:3*units]) # Candidate
        o = sigmoid(z[3*units:])
        
        c = f * c + i * g
        h = o * np.tanh(c)
        
    # 3. Dense Layer 1 (ReLU)
    w1 = weights['dense_1_kernel']
    b1 = weights['dense_1_bias']
    x = relu(np.dot(h, w1) + b1)
    
    # 4. Dense Layer 2 (Output Sigmoid)
    w2 = weights['dense_2_kernel']
    b2 = weights['dense_2_bias']
    prediction = sigmoid(np.dot(x, w2) + b2)[0]
    
    return prediction

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

    # Use NumPy for prediction
    prediction = predict_numpy(combined_text)

    result = "Fraudulent 🚨" if prediction > 0.7 else "Legitimate ✅"
    print(f"Prediction result: {result} (Score: {prediction:.4f})")

    return render_template(
        "index.html",
        prediction=f"The job post is: {result}"
    )

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
