import numpy as np
import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Computes the partial derivative of an output with respect to an input.

    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor

    Returns:
        Derivatives of outputs w.r.t. inputs.
    """
    # It is essential to set create_graph to True to be able to manipulate the gradients afterward
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


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
