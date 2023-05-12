from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.components.mlp import MLP
from src.models.pinns_cooling_law import PINNs_cooling_law

# Set device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
np.random.seed(10)
torch.manual_seed(42)


# Create data
def cooling_law(time: float, Tenv: float, T0: float, k: float) -> float:
    """Compute the temperature at a given time using the parameters provided.

    Args:
        time: Time variable
        Tenv: Ambient temperature (constant).
        T0: Initial temperature (constant).
        k: Decay (constant).

    Returns:
        Temperature, T at a given time.
    """
    T = Tenv + (T0 - Tenv) * np.exp(-k * time)
    return T


# Define the cooling parameters
Tenv = 25  # Ambient temperature
T0 = 100  # Starting temperature
k = 0.005  # Decay constant

# Define the full time axis of interest
times = np.linspace(0, 1000, 1000)

# Define the corresponding cooling law
eq = partial(cooling_law, Tenv=Tenv, T0=T0, k=k)

# Compute temperature values based on the pre-defined cooling law
temperatures = eq(times)

# Make measured data points with randomness
time_samples = np.linspace(0, 300, 10)
temperature_samples = eq(time_samples) + 2 * np.random.randn(10)

# Display the generated input that characterize the problem
plt.figure(1)
plt.plot(times, temperatures)
plt.plot(time_samples, temperature_samples, "o")
plt.legend(["Cooling law", "Measured data"])
plt.ylabel("Temperature (C)")
plt.xlabel("Time (s)")

# Create an MLP network
mlp = MLP([1], [1], [100, 100, 100, 100]).to(DEVICE)

# Define optimizer, learning rate, and number of epochs
lr = 1e-5
optimizer = partial(torch.optim.Adam, lr=lr)
epochs = 20000

# Create the PINNs model
pinn = PINNs_cooling_law(mlp, epochs=epochs, optimizer=optimizer)
losses = pinn.optimize(time_samples, temperature_samples, DEVICE)

# Display the evolution of the loss over the epoch
# to check the good behavior of this phase, i.e., the loss must decrease
plt.figure(2)
plt.plot(losses)
plt.yscale("log")

# Use the NN to predict the temperatures at the full range time axis
temperature_predictions = pinn.predict(times, DEVICE)

# Display the predicted results
plt.figure(3)
plt.plot(times, temperatures, alpha=0.8)
plt.plot(time_samples, temperature_samples, "o")
plt.plot(times, temperature_predictions, alpha=0.8)
plt.legend(labels=["Cooling law", "Measured data", "Network"])
plt.ylabel("Temperature (C)")
plt.xlabel("Time (s)")
