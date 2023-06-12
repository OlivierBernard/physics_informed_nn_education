import torch


def grad(
    outputs: torch.Tensor, inputs: torch.Tensor, order: int = 1, create_graph: bool = False
) -> torch.Tensor:
    """Computes the first order or second order partial derivative of an output with respect to an
    input.

    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor

    Returns:
        Derivatives of outputs w.r.t. inputs.
    """
    match order:
        case 1:
            return torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=torch.ones_like(outputs),
                retain_graph=True,
                create_graph=create_graph,
            )[0]
        case 2:
            first_order = grad(outputs, inputs, order=order - 1, create_graph=True)
            return grad(first_order, inputs, order=order - 1, create_graph=False)
        case _:
            raise NotImplementedError()
