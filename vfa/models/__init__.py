import os
import importlib.util
import pdb

# Directory containing network files
models_directory = os.path.dirname(__file__)


def create_network_class(network_name):
    """
    Create a class of the specified network.

    Args:
        network_name (str): Name of the network.

    Returns:
        network_class: Class of the network.

    Raises:
        FileNotFoundError: If the network file is not found.
        AttributeError: If the network class is not found.
    """

    network_file_path = os.path.join(models_directory, f"{network_name.lower()}.py")
    if not os.path.exists(network_file_path):
        raise FileNotFoundError(f"Network file '{network_name.lower()}.py' not found in '{models_directory}'.")

    network_module_name = f"models.{network_name.lower()}"
    network_module_spec = importlib.util.spec_from_file_location(network_module_name, network_file_path)
    network_module = importlib.util.module_from_spec(network_module_spec)
    network_module_spec.loader.exec_module(network_module)

    network_class = getattr(network_module, network_name, None)

    if network_class is None:
        raise AttributeError(f"Network class '{network_name}' not found in module '{network_module_name}'.")

    return network_class
