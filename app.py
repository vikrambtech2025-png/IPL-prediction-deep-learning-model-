import streamlit as st
import matplotlib.pyplot as plt
import torch
import os

from preprocess import load_data, scale_data, create_sequences
from model import TransformerModel

st.set_page_config(page_title="IPL Predictor", layout="wide")

st.title("🏏 IPL Score Predictor (Transformer AI)")

# Load data
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Plot
st.subheader("Score Trend")

fig, ax = plt.subplots()
ax.plot(df['total_score'])
ax.set_xlabel("Ball Index")
ax.set_ylabel("Score")
st.pyplot(fig)

# Load model
model = TransformerModel()

model_path = os.path.join("models", "saved_model.pth")

if not os.path.exists(model_path):
    st.error("Model not found! Run train.py first.")
    st.stop()

model.load_state_dict(torch.load(model_path))
model.eval()

# Prepare data
scaled, scaler = scale_data(df)
X, y = create_sequences(scaled)

# Predict
st.subheader("Predict Next Score")

if st.button("Predict"):

    with st.spinner("Predicting..."):

        seq = X[-1]
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(seq).item()

        result = scaler.inverse_transform([[0,0,0,pred]])[0][-1]

    st.success(f"Predicted Score: {int(result)}")