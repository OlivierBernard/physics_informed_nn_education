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
    ) -> None:
        """Initialize class instance.

        Args:
            mlp: MLP network.
            epochs: Number of epochs.
            loss: Reconstruction loss function.
            optimizer: Optimizer.
            loss2: Optional regularization term.
            loss2_weight: Regularization weight.
            physical_loss: Whether to use physical loss.
        """
        super().__init__()

        # Define attributes
        self.epochs = epochs
        self.loss = loss
        self.mlp = mlp
        self.optimizer = optimizer(self.mlp.parameters())
        self.lambda_div = 1  # Weight for divergence residual
        self.lambda_BC = 1  # Weight for boundary residual
        self.do_phy_loss = physical_loss

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
        grid_R = torch.from_numpy(batch["grid_R"]).float().requires_grad_(True).to(device)
        grid_TH = torch.from_numpy(batch["grid_TH"]).float().requires_grad_(True).to(device)

        x = torch.stack((grid_R, grid_TH), dim=1)

        # Compute the corresponding estimated velocities
        preds = self.forward(x)
        uR = preds[:, :-1]
        uTH = preds[:, -1:]

        # Compute the divergence
        rdiv = grid_R * grad(uR, grid_R, order=1) + uR + grad(uTH, grid_TH, order=1)

        # Compute residual of boundary points
        grid_R_wall = torch.from_numpy(batch["grid_R_wall"]).float().to(device)
        grid_TH_wall = torch.from_numpy(batch["grid_TH_wall"]).float().to(device)
        x_wall = torch.stack((grid_R_wall, grid_TH_wall), dim=1)

        nR_wall = torch.from_numpy(batch["nR_wall"]).float().to(device)
        nTH_wall = torch.from_numpy(batch["nTH_wall"]).float().to(device)
        uR_wall_ref = torch.from_numpy(batch["uR_wall"]).float().to(device)
        uTH_wall_ref = torch.from_numpy(batch["uTH_wall"]).float().to(device)

        u_wall = self.forward(x_wall)
        uR_wall = u_wall[:, :-1]
        uTH_wall = u_wall[:, -1:]

        res_BC = (uR_wall - uR_wall_ref) * nR_wall + (uTH_wall - uTH_wall_ref) * nTH_wall

        return torch.mean(torch.abs(rdiv)), torch.mean(torch.abs(res_BC))

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
        u_ref = self.np_to_th(np.stack((batch["uR"], batch["uTH"]), axis=1), device)

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
            # Apply the forward pass
            outputs = self.forward(x)
            # Compute the corresponding loss value based on the predicted outputs
            loss = self.loss(outputs, u_ref)
            # Update the loss value if an additional loss is defined
            if self.do_phy_loss:
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
            fit_pbar.set_postfix(pbar_metrics)
            # Store the value of the loss function
            losses.append(final_loss.item())
            # Display loss value each round(self.epochs/10) epochs
        # Return the losses list at the end of the main loop
        return losses

    def predict(self, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Prediction loop."""
        x = self.np_to_th(np.stack((batch["grid_R"], batch["grid_TH"]), axis=1), device)
        # Put torch layers (batchnorm/dropout etc.) in evaluation mode.
        self.eval()

        # Disable gradient flow
        with torch.no_grad():
            # Apply the forward pass
            out = self.forward(x)
        # Return the numpy prediction detached from the computation graph
        return out.detach().cpu().numpy()
