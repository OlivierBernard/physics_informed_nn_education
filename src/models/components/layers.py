from torch import nn


def get_nn_module(module: str, *module_args, **module_kwargs) -> nn.Module:
    """Instantiates an ``nn.Module`` with the requested parameters.

    Args:
        module: Name of the ``nn.Module`` to instantiate.
        *module_args: Positional arguments to pass to the ``nn.Module``'s constructor.
        **module_kwargs: Keyword arguments to pass to the ``nn.Module``'s constructor.

    Returns:
        Instance of the ``nn.Module``.
    """
    return getattr(nn, module)(*module_args, **module_kwargs)
