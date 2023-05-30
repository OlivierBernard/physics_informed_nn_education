import os
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from einops.einops import rearrange
from mat73 import loadmat

from src.models.components.mlp import MLP
from src.models.pinns_ivfm import PINNs_ivfm

# Set device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
np.random.seed(12345)
torch.manual_seed(12345)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12345)

# Create data
# Set current working directory
cwd = Path(__file__).resolve().parent
os.chdir(cwd)
patient = "../../data/patient_06_1.mat"
data = loadmat(patient)
fname = os.path.basename(patient)[:-4]
spacing = [0.5, 0.5, 1]

# Create log folder
save_dir = os.path.join("../../logs", "iVFM")
os.makedirs(save_dir, exist_ok=True)

# Weights for losses
lambda_div = 1e-4
lambda_BC = 1
lambda_smooth = 1e-5

nb_frame = data["uR"].shape[-1]
preds_array = np.zeros([2, *data["uR"].shape])

# nb_frame = 1

# Create an MLP network
mlp = MLP([2], [2], [100, 100, 100, 100, 100, 100, 100, 100], activation="tanh")

# Define optimizer, learning rate, and number of epochs
lr = 1e-4
optimizer = partial(torch.optim.Adam, lr=lr)
# optimizer = partial(torch.optim.LBFGS, lr=2, max_iter=100000)
epochs = 1000

# Create the PINNs model
pinn = PINNs_ivfm(
    mlp,
    epochs=epochs,
    optimizer=optimizer,
    lambda_div=lambda_div,
    lambda_BC=lambda_BC,
    lambda_smooth=lambda_smooth,
).to(DEVICE)

nb_frame = 1
for frame in range(nb_frame):
    # Get radial and angular velocity
    VNyquist = data["VNyquist"]
    powers = np.log10(data["powers"][..., frame]) / 2
    uD = data["uD"][..., frame] * VNyquist
    uR = data["uR"][..., frame]
    uTH = data["uTH"][..., frame]

    # Get left ventricle mask
    lv_seg = (~np.isnan(uR)).astype(np.uint8)

    # Get boundary points' coordinates
    nR_wall = data["nR_wall"][..., frame]
    nTH_wall = data["nTH_wall"][..., frame]

    # Get boundary points' velocities
    uR_wall = data["uR_wall"][..., frame]
    uTH_wall = data["uTH_wall"][..., frame]

    # Get sampling grid
    grid_R = data["grid_R"]
    grid_TH = data["grid_TH"]

    # Extract data points in the left ventricle
    grid_R_s = grid_R[lv_seg == 1]
    grid_TH_s = grid_TH[lv_seg == 1]
    powers_s = powers[lv_seg == 1]
    uD_s = uD[lv_seg == 1]
    uR_s = uR[lv_seg == 1]
    uTH_s = uTH[lv_seg == 1]
    nR_wall_s = nR_wall[nR_wall != 0]
    nTH_wall_s = nTH_wall[nTH_wall != 0]
    uD_wall_s = uD[nR_wall != 0]
    uR_wall_s = uR_wall[nR_wall != 0]
    uTH_wall_s = uTH_wall[nR_wall != 0]
    grid_R_wall_s = grid_R[nR_wall != 0]
    grid_TH_wall_s = grid_TH[nTH_wall != 0]

    # Store data in a dict
    batch = {}
    batch["powers"] = powers_s
    batch["grid_R"] = grid_R_s
    batch["grid_TH"] = grid_TH_s
    batch["uD"] = uD_s
    batch["uR"] = uR_s
    batch["uTH"] = uTH_s
    batch["grid_R_wall"] = grid_R_wall_s
    batch["grid_TH_wall"] = grid_TH_wall_s
    batch["uD_wall"] = uD_wall_s
    batch["uR_wall"] = uR_wall_s
    batch["uTH_wall"] = uTH_wall_s
    batch["nR_wall"] = nR_wall_s
    batch["nTH_wall"] = nTH_wall_s

    losses = pinn.optimize(batch, DEVICE)

    # Inference
    u_pred = pinn.predict(batch, DEVICE)
    uR_pred = u_pred[:, 0]
    uTH_pred = u_pred[:, 1]
    preds_array[0, ..., frame][lv_seg == 1] = uR_pred
    preds_array[1, ..., frame][lv_seg == 1] = uTH_pred

    pinn.epochs = epochs / 2

# Save predicted velocities
itk_image = sitk.GetImageFromArray(rearrange(preds_array, "c h w d ->  d h w c"))
itk_image.SetSpacing(spacing)
sitk.WriteImage(itk_image, os.path.join(save_dir, fname + "_test1.nii.gz"))
