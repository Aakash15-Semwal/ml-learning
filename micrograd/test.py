import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from basic_net import basic_MLP
from engine import Value

sns.set_theme(style="darkgrid")

# Generate Training Data (Randomly sampled across the domain)
def generate_decay_data(num_samples=150):
    dataset = []
    for _ in range(num_samples):
        t = random.uniform(0, 6)
        y_true = math.exp(-0.5 * t)
        dataset.append(([t], y_true))
    return dataset

# Generate Dense Evaluation Data (Ordered sequentially for clean line plotting)
def generate_eval_data(num_samples=100):
    t_eval = np.linspace(0, 6, num_samples)
    y_eval = np.exp(-0.5 * t_eval)
    return t_eval, y_eval

# Initialize data
train_data = generate_decay_data(150)
t_eval, y_actual = generate_eval_data(100)

# Architecture: 1 Input -> Hidden Layer 1 (8 neurons) -> Hidden Layer 2 (4 neurons) -> 1 Output
mlp = basic_MLP(input_size=1, layers=[8, 4], output_size=1)
epochs = 100
learning_rate = 0.25

print("Starting training...")
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    random.shuffle(train_data)
    
    for x_raw, y_raw in train_data:
        x = [Value(x_raw[0])]
        y_true = Value(y_raw)
        
        # Forward pass
        y_pred = mlp(x)
        
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        
        # Loss calculation: (y_pred - y_true)^2
        error = y_pred + (y_true * -1)
        loss = error * error
        epoch_loss += loss.data
        
        # Backward pass
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()
        
        # Gradient descent step
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
            
    # Calculate average loss for the dataset pass
    avg_loss = epoch_loss / len(train_data)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | Average MSE Loss: {avg_loss:.6f}")

print("Training complete!")

# Generate Predictions from the Trained Network
y_predicted = []
for t_val in t_eval:
    # Forward pass on evaluation points without tracking gradients
    pred = mlp([Value(t_val)])
    if isinstance(pred, list):
        pred = pred[0]
    y_predicted.append(pred.data)

y_predicted = np.array(y_predicted)

# 6. Plotting the Comparison using Seaborn and Matplotlib
plt.figure(figsize=(10, 6))

# Plot the true mathematical curve
sns.lineplot(x=t_eval, y=y_actual, label="Original Function ($e^{-0.5t}$)", color="royalblue", linewidth=2.5)

# Plot the neural network's approximation
sns.lineplot(x=t_eval, y=y_predicted, label="Neural Net Prediction", color="crimson", linestyle="--", linewidth=2.5)

plt.title("Neural Network Approximation vs. Original Function after 100 Epochs", fontsize=14, pad=15)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("Value (y)", fontsize=12)
plt.legend(fontsize=11)

# Display the plot
plt.show()

# Calculate residual differences
residuals = y_actual - y_predicted

plt.figure(figsize=(10, 4))
# Plot the distribution of errors
sns.histplot(residuals, kde=True, color="purple", bins=15)
plt.axvline(x=0, color="black", linestyle=":")
plt.title("Distribution of Prediction Errors (Residuals)", fontsize=12)
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Count")
plt.show()