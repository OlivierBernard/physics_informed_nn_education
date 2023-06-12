from functools import partial

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.utils.automatic_differentiation import grad


class PINNs_cooling_law(nn.Module):
    """Neural network with an MLP architecture."""

    # Define the constructor class with inheritance
    def __init__(
        self,
        mlp: nn.Module,
        epochs: int = 1000,
        loss: nn.Module = nn.MSELoss(),
        optimizer: torch.optim.Optimizer = partial(torch.optim.Adam, lr=1e-3),
        loss2: nn.Module = None,
        loss2_weight: float = 1,
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
        self.loss2_weight = loss2_weight
        self.mlp = mlp
        self.optimizer = optimizer(self.mlp.parameters())
        self.Tenv = 25  # Ambient temperature
        self.k = 0.005  # Decay constant

        if not loss2 and physical_loss:
            self.loss2 = partial(self.physical_loss)
        else:
            self.loss2 = loss2

    def np_to_th(self, x: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy data to torch.Tensor with the selected DEVICE (cpu or gpu).

        Args:
            x: Numpy data.

        Returns:
            Converted torch.Tensor data.
        """
        n_samples = len(x)
        return torch.from_numpy(x).to(torch.float).to(device).reshape(n_samples, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)

    def physical_loss(self, x: torch.Tensor, device) -> torch.Tensor:
        """Compute physical loss."""
        # Define the full time axis of interest
        # Set requires_grad to True to be able to compute the gradients w.r.t. input
        x = x.requires_grad_(True).to(device)
        # Compute the corresponding estimated temperature
        temperatures_full = self.forward(x)
        # Compute the corresponding gradients
        dT = grad(temperatures_full, x, order=1)
        # Compute residual loss for each temperature term
        residual = dT - self.k * (self.Tenv - temperatures_full)
        # Return residual loss
        return torch.mean(residual**2)

    def optimize(self, X: np.ndarray, y: np.ndarray, device: torch.device) -> list[float]:
        """Training loop.

        Args:
            X: Input data.
            y: Reference.

        Returns:
            List of losses over epochs.
        """
        # Convert numpy to torch Tensor
        Xt = self.np_to_th(X, device)
        yt = self.np_to_th(y, device)

        times_full = torch.linspace(
            0,
            1000,
            steps=1000,
        ).view(-1, 1)

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
            outputs = self.forward(Xt)
            # Compute the corresponding loss value based on the predicted outputs
            loss = self.loss(yt, outputs)
            # Update the loss value if an additional loss is defined
            if self.loss2:
                loss += self.loss2_weight * self.loss2(times_full, device)
            # Compute the backward pass
            # This step will compute the NN gradient thank to autograd concepts
            loss.backward()
            # Update the NN parameters based on the NN gradient previously computed
            # and the optimizer scheme (in this example adam)
            self.optimizer.step()
            pbar_metrics["train_loss"] = loss.item()
            fit_pbar.set_postfix(pbar_metrics)
            # Store the value of the loss function
            losses.append(loss.item())
            # Display loss value each round(self.epochs/10) epochs
        # Return the losses list at the end of the main loop
        return losses

    def predict(self, X: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Prediction loop."""
        # Put torch layers (batchnorm/dropout etc.) in evaluation mode.
        self.eval()

        # Disable gradient flow
        with torch.no_grad():
            # Apply the forward pass
            out = self.forward(self.np_to_th(X, device))
        # Return the numpy prediction detached from the computation graph
        return out.detach().cpu().numpy()
