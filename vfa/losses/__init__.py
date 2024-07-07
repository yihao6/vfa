import os
import importlib.util

# Directory containing loss files
losses_directory = os.path.dirname(__file__)


def create_loss_instance(loss_name, **kwargs):
    """
    Create an instance of the specified loss class with optional parameters.

    Args:
        loss_name (str): Name of the loss class.
        **kwargs: Optional parameters to initialize the loss class.

    Returns:
        loss_instance: Instance of the loss class.

    Raises:
        FileNotFoundError: If the loss file is not found.
        AttributeError: If the loss class is not found or cannot be instantiated.
    """

    loss_file_path = os.path.join(losses_directory, f"{loss_name.lower()}_loss.py")
    if not os.path.exists(loss_file_path):
        raise FileNotFoundError(f"Loss file '{loss_name.lower()}_loss.py' not found in '{losses_directory}'.")

    loss_module_name = f"losses.{loss_name.lower()}_loss"
    loss_module_spec = importlib.util.spec_from_file_location(loss_module_name, loss_file_path)
    loss_module = importlib.util.module_from_spec(loss_module_spec)
    loss_module_spec.loader.exec_module(loss_module)

    loss_class = getattr(loss_module, loss_name, None)

    if loss_class is None:
        raise AttributeError(f"Loss class '{loss_name}' not found in module '{loss_module_name}'.")

    loss_instance = loss_class(**kwargs)
    return loss_instance
