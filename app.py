import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("📊 ML Assignment-2: Linear Regression")
st.write("**Author:** Md. Shakeeb Ashraf")

# Dataset
X = np.array([1, 2, 4, 6, 8])
Y = np.array([3, 5, 9, 13, 17])

st.subheader("📌 Dataset")
st.write({"Operating Hours (X)": X, "Consumption (Y)": Y})

# User inputs
w0 = st.slider("Bias (w0)", -10.0, 10.0, 2.0)
w1 = st.slider("Weight (w1)", -10.0, 10.0, 2.0)

# Prediction
Y_pred = w0 + w1 * X

# Plot
fig, ax = plt.subplots()
ax.scatter(X, Y, label="Actual Data")
ax.plot(X, Y_pred, color="red", label="Predicted Line")
ax.set_xlabel("Operating Hours")
ax.set_ylabel("Consumption")
ax.legend()

st.subheader("📈 Regression Result")
st.pyplot(fig)

# Loss
loss = np.mean((Y - Y_pred) ** 2)
st.success(f"Mean Squared Error: {loss:.2f}")
