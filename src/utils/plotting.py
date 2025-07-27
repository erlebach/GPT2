import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the metrics CSV file
df = pd.read_csv("lightning_logs/version_25/metrics.csv")

# Extract training and validation loss data
# Training loss is in 'train_loss_step' column
# Validation loss is in 'val_loss' column
# Steps are in 'step' column

# Filter out rows with NaN values
train_data = df[["step", "train_loss_step"]].dropna()
val_data = df[["step", "val_loss"]].dropna()

# Create the plot
plt.figure(figsize=(12, 8))

# Plot training loss
plt.plot(
    train_data["step"],
    train_data["train_loss_step"],
    label="Training Loss",
    color="blue",
    alpha=0.7,
    linewidth=1.5,
)

# Plot validation loss
plt.plot(
    val_data["step"],
    val_data["val_loss"],
    label="Validation Loss",
    color="red",
    alpha=0.7,
    linewidth=1.5,
)

# Customize the plot
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title(
    "Training and Validation Loss Over Time\n(Showing Clear Overfitting Pattern)",
    fontsize=16,
    fontweight="bold",
)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add annotations to highlight the overfitting
plt.annotate(
    "Training Loss\nDecreasing",
    xy=(train_data["step"].iloc[-1], train_data["train_loss_step"].iloc[-1]),
    xytext=(
        train_data["step"].iloc[-1] - 1000,
        train_data["train_loss_step"].iloc[-1] - 2,
    ),
    arrowprops=dict(arrowstyle="->", color="blue", alpha=0.7),
    fontsize=10,
    color="blue",
)

plt.annotate(
    "Validation Loss\nIncreasing\n(Overfitting!)",
    xy=(val_data["step"].iloc[-1], val_data["val_loss"].iloc[-1]),
    xytext=(val_data["step"].iloc[-1] - 1000, val_data["val_loss"].iloc[-1] + 2),
    arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
    fontsize=10,
    color="red",
)

# Add a shaded region to show the overfitting gap
plt.fill_between(
    val_data["step"],
    val_data["val_loss"],
    train_data["train_loss_step"].iloc[: len(val_data)],
    alpha=0.2,
    color="orange",
    label="Overfitting Gap",
)

# Set y-axis limits to focus on the relevant range
plt.ylim(0, 12)

# Add statistics
train_final = train_data["train_loss_step"].iloc[-1]
val_final = val_data["val_loss"].iloc[-1]
gap = val_final - train_final

plt.text(
    0.02,
    0.98,
    f"Final Training Loss: {train_final:.3f}\nFinal Validation Loss: {val_final:.3f}\nOverfitting Gap: {gap:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

plt.tight_layout()
plt.show()

# Also create a zoomed-in version focusing on the early training
plt.figure(figsize=(12, 8))

# Focus on first 1000 steps
early_train = train_data[train_data["step"] <= 1000]
early_val = val_data[val_data["step"] <= 1000]

plt.plot(
    early_train["step"],
    early_train["train_loss_step"],
    label="Training Loss",
    color="blue",
    alpha=0.7,
    linewidth=1.5,
)
plt.plot(
    early_val["step"],
    early_val["val_loss"],
    label="Validation Loss",
    color="red",
    alpha=0.7,
    linewidth=1.5,
)

plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title(
    "Early Training Phase (First 1000 Steps)\n(When Overfitting Started)",
    fontsize=16,
    fontweight="bold",
)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Highlight where overfitting begins
overfitting_start = early_val[early_val["val_loss"] > early_val["val_loss"].iloc[0]]
if len(overfitting_start) > 0:
    plt.axvline(
        x=overfitting_start["step"].iloc[0],
        color="orange",
        linestyle="--",
        label=f'Overfitting Starts (~{overfitting_start["step"].iloc[0]} steps)',
    )
    plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
