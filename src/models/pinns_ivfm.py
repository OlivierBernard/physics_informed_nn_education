from functools import partial

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.utils.automatic_differentiation import grad


class PINNs_ivfm(nn.Module):
    """Neural network with an MLP architecture."""

    # Define the constructor class with inheritance
    def __init__(
        self,
        mlp: nn.Module,
        epochs: int = 1000,
        loss: nn.Module = nn.L1Loss(),
        optimizer: torch.optim.Optimizer = partial(torch.optim.Adam, lr=1e-3),
        physical_loss: bool = True,
        lambda_div: float = 1,
        lambda_BC: float = 1,
        lambda_smooth: float = 1,
    ) -> None:
        """Initialize class instance.

        Args:
            mlp: MLP network.
            epochs: Number of epochs.
            loss: Reconstruction loss function.
            optimizer: Optimizer.
            physical_loss: Whether to use physical loss.
            lambda_div: Weight for free divergence residual loss.
            lambda_BC: Weight for boundary condition residual loss.
            lambda_smooth: Weight for smoothing regularization.
        """
        super().__init__()

        # Define attributes
        self.epochs = epochs
        self.loss = loss
        self.mlp = mlp
        self.lambda_div = lambda_div  # Weight for divergence residual
        self.lambda_BC = lambda_BC  # Weight for boundary residual
        self.lambda_smooth = lambda_smooth  # Weight for smoothing regularization
        self.do_phy_loss = physical_loss
        self.optimizer = optimizer(self.mlp.parameters())
        if isinstance(self.lambda_div, torch.nn.parameter.Parameter):
            self.optimizer.add_param_group({"params": self.lambda_div})
        if isinstance(self.lambda_BC, torch.nn.parameter.Parameter):
            self.optimizer.add_param_group({"params": self.lambda_BC})
        if isinstance(self.lambda_smooth, torch.nn.parameter.Parameter):
            self.optimizer.add_param_group({"params": self.lambda_smooth})
        self.relu = torch.nn.ReLU(inplace=False)

    def np_to_th(self, x: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy data to torch.Tensor with the selected DEVICE (cpu or gpu).

        Args:
            x: Numpy data.

        Returns:
            Converted torch.Tensor data.
        """
        n_samples = len(x)
        return torch.from_numpy(x).float().to(device).reshape(n_samples, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)

    def physical_loss(self, batch: dict[str, torch.Tensor], device) -> torch.Tensor:
        """Compute physical loss."""
        grid_R = self.np_to_th(batch["grid_R"], device).requires_grad_(True)
        grid_TH = self.np_to_th(batch["grid_TH"], device).requires_grad_(True)

        x = torch.hstack((grid_R, grid_TH))

        # Compute the corresponding estimated velocities
        preds = self.forward(x)
        uR = preds[:, :-1]
        uTH = preds[:, -1:]

        # Compute the divergence
        uR_dr = grad(uR, grid_R, order=1, create_graph=True)
        uTH_dth = grad(uTH, grid_TH, order=1, create_graph=True)
        rdiv = grid_R * uR_dr + uR + uTH_dth

        # Smoothing regularization
        if self.lambda_smooth:
            uR_dr_dr = grad(uR_dr, grid_R, order=1, create_graph=True)
            uTH_dth_dth = grad(uTH_dth, grid_TH, order=1, create_graph=True)
            uTH_dr_dr = grad(uTH, grid_R, order=2)
            uR_dth_dth = grad(uR, grid_TH, order=2)
            uR_dth = grad(uR, grid_TH, order=1, create_graph=True)
            uR_dth_dr = grad(uR_dth, grid_R, order=1, create_graph=True)
            uTH_dth_dr = grad(uTH_dth, grid_R, order=1, create_graph=True)

            res_smooth = (
                (grid_R * grid_R * uR_dr_dr) ** 2
                + 2 * (grid_R * uR_dth_dr) ** 2
                + uR_dth_dth**2
                + (grid_R * grid_R * uTH_dr_dr) ** 2
                + 2 * (grid_R * uTH_dth_dr) ** 2
                + uTH_dth_dth**2
            )

        # Compute residual of boundary points
        grid_R_wall = self.np_to_th(batch["grid_R_wall"], device).requires_grad_(True)
        grid_TH_wall = self.np_to_th(batch["grid_TH_wall"], device).requires_grad_(True)
        # uD_wall = self.np_to_th(batch["uD_wall"], device)
        # x_wall = torch.hstack((grid_R_wall, grid_TH_wall, uD_wall))
        x_wall = torch.hstack((grid_R_wall, grid_TH_wall))

        nR_wall = self.np_to_th(batch["nR_wall"], device).requires_grad_(True)
        nTH_wall = self.np_to_th(batch["nTH_wall"], device).requires_grad_(True)
        uR_wall_ref = self.np_to_th(batch["uR_wall"], device).requires_grad_(True)
        uTH_wall_ref = self.np_to_th(batch["uTH_wall"], device).requires_grad_(True)

        u_wall = self.forward(x_wall)
        uR_wall = u_wall[:, :-1]
        uTH_wall = u_wall[:, -1:]

        res_BC = (uR_wall - uR_wall_ref) * nR_wall + (uTH_wall - uTH_wall_ref) * nTH_wall

        if self.lambda_smooth:
            return (
                self.loss(rdiv, torch.zeros_like(rdiv)),
                self.loss(res_BC, torch.zeros_like(res_BC)),
                self.loss(res_smooth, torch.zeros_like(res_smooth)),
            )
        else:
            return self.loss(rdiv, torch.zeros_like(rdiv)), self.loss(
                res_BC, torch.zeros_like(res_BC)
            )

    def optimize(self, batch: dict[str, torch.Tensor], device: torch.device) -> list[float]:
        """Training loop.

        Args:
            X: Input data.
            y: Reference.

        Returns:
            List of losses over epochs.
        """
        # Prepare training input and reference
        x = self.np_to_th(np.stack((batch["grid_R"], batch["grid_TH"]), axis=1), device)
        uD = self.np_to_th(batch["uD"], device)
        # powers = self.np_to_th(batch["powers"], device)

        # Put torch layers (batchnorm/dropout etc.) in training mode.
        self.train()
        losses = []
        # Main iteration loop to optimize the MLP parameters
        fit_pbar = tqdm(range(self.epochs), desc="Optimization", unit="epoch")
        pbar_metrics = {"train_loss": None}
        for ep in fit_pbar:
            # At each iteration, we need to set the computed gradient to zero
            # since the call to the backward function will automatically compute
            # the NN gradients
            self.optimizer.zero_grad()
            if isinstance(self.lambda_div, torch.nn.parameter.Parameter):
                self.lambda_div.data = self.relu(self.lambda_div)
            if isinstance(self.lambda_BC, torch.nn.parameter.Parameter):
                self.lambda_BC.data = self.relu(self.lambda_BC)
            if self.lambda_smooth and isinstance(self.lambda_smooth, torch.nn.parameter.Parameter):
                self.lambda_smooth.data = self.relu(self.lambda_smooth)
            # Apply the forward pass
            outputs = self.forward(x)
            # Compute the reconstruction loss
            loss = self.loss(outputs[:, :-1], -uD)
            # Update the loss value if additional losses are requested
            if self.do_phy_loss:
                if self.lambda_smooth:
                    res_div, res_BC, res_smooth = self.physical_loss(batch, device)
                    final_loss = (
                        loss
                        + self.lambda_div * res_div
                        + self.lambda_BC * res_BC
                        + self.lambda_smooth * res_smooth
                    )
                else:
                    res_div, res_BC = self.physical_loss(batch, device)
                    final_loss = loss + self.lambda_div * res_div + self.lambda_BC * res_BC
            else:
                final_loss = loss
            # Compute the backward pass
            # This step will compute the NN gradient thank to autograd concepts
            final_loss.backward()
            # Update the NN parameters based on the NN gradient previously computed
            # and the optimizer scheme (in this example adam)
            self.optimizer.step()
            pbar_metrics["train_loss"] = final_loss.item()
            pbar_metrics["res_recon"] = loss.item()
            pbar_metrics["residual_div"] = res_div.item()
            pbar_metrics["residual_BC"] = res_BC.item()
            if self.lambda_smooth:
                pbar_metrics["residual_smooth"] = res_smooth.item()
            fit_pbar.set_postfix(pbar_metrics)
            # Store the value of the loss function
            losses.append(final_loss.item())
            # Display loss value each round(self.epochs/10) epochs
        # Return the losses list at the end of the main loop
        return losses

    def predict(self, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Prediction loop."""
        # x = self.np_to_th(np.stack((batch["grid_R"], batch["grid_TH"], batch["uD"]), axis=1), device)
        x = self.np_to_th(np.stack((batch["grid_R"], batch["grid_TH"]), axis=1), device)
        # Put torch layers (batchnorm/dropout etc.) in evaluation mode.
        self.eval()

        # Disable gradient flow
        with torch.no_grad():
            # Apply the forward pass
            out = self.forward(x)
        # Return the numpy prediction detached from the computation graph
        return out.detach().cpu().numpy()
