import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set title
st.title("🔢 Linear Regression — Math Visualization (Gradient Descent)")

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature: weight (e.g., 0 to 10 kg)
true_w = 2.5
true_b = 5
y = true_w * X + true_b + np.random.randn(100, 1) * 2  # Add noise

# Scatter plot
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, color='blue', label='Data')
ax1.set_xlabel("x (Weight)")
ax1.set_ylabel("y (Price)")
ax1.set_title("Generated Data")
st.pyplot(fig1)

# Linear Regression function
def predict(X, w, b):
    return w * X + b

# Loss: Mean Squared Error
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient computation
def compute_gradients(X, y, w, b):
    n = len(X)
    y_pred = predict(X, w, b)
    dw = (-2 / n) * np.sum(X * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)
    return dw, db

# Sidebar: Hyperparameters
st.sidebar.title("⚙️ Gradient Descent Controls")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001)
iterations = st.sidebar.slider("Iterations", 1, 200, 50, 1)

# Initialize w and b
w = 0.0
b = 0.0
loss_history = []

# Perform gradient descent
for i in range(iterations):
    y_pred = predict(X, w, b)
    loss = compute_loss(y, y_pred)
    loss_history.append(loss)

    dw, db = compute_gradients(X, y, w, b)
    w -= learning_rate * dw
    b -= learning_rate * db

    # Optional animation (one step)
    if i % (iterations // 10) == 0 or i == iterations - 1:
        st.write(f"Iteration {i + 1}: Loss = {loss:.4f}, w = {w:.3f}, b = {b:.3f}")

# Final prediction line
y_final = predict(X, w, b)

# Plot final fit
fig2, ax2 = plt.subplots()
ax2.scatter(X, y, label='Data')
ax2.plot(X, y_final, color='red', label=f'Prediction Line: y = {w:.2f}x + {b:.2f}')
ax2.legend()
ax2.set_title("Final Model Fit")
st.pyplot(fig2)

# Plot Loss over iterations
fig3, ax3 = plt.subplots()
ax3.plot(range(1, iterations + 1), loss_history, color='green')
ax3.set_title("Loss over Iterations")
ax3.set_xlabel("Iteration")
ax3.set_ylabel("MSE Loss")
st.pyplot(fig3)
