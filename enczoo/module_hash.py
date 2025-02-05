import hashlib

import numpy as np
import torch


def hash_torch_module(module: torch.nn.Module) -> str:
    # Hash the model's state dict
    state_dict = module.state_dict()
    ordered_keys = sorted(state_dict.keys())
    tensor_hashes = [hash_tensor(tensor=state_dict[key]) for key in ordered_keys]

    # Hash the model's string representation
    architecture_hash = hash_string(input_string=str(module))

    # Combine the hashes
    model_hash = hash_string(input_string=architecture_hash + '.' + ''.join(tensor_hashes))

    # Write the model hash
    return model_hash


def hash_tensor(tensor: torch.Tensor) -> str:
    """
    Hashes a torch.Tensor.
    :param tensor:
    :return:
    """
    # Convert the tensor to a numpy array
    tensor_data = tensor.detach().cpu().numpy()

    return hash_ndarray(x=tensor_data)


def hash_ndarray(x: np.ndarray) -> str:
    """
    Hash an ndarray.
    :param x:
    :return:
    """

    sha256_hash = hashlib.sha256()

    # Update the hash with the ndarray data
    sha256_hash.update(x.tobytes())

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


def hash_string(input_string: str) -> str:
    sha256_hash = hashlib.sha256()

    # Convert string to bytes
    sha256_hash.update(input_string.encode('utf-8'))

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()
