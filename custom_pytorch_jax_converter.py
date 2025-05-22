import torch
import numpy as np
from flax.core import freeze, unfreeze

# Load PyTorch state_dict
state_dict = torch.load("/results/pytorch_base_model_criteo1tb_22_may.pth")

# Convert PyTorch tensors to NumPy arrays
numpy_weights = {k: v.numpy() for k, v in state_dict.items()}


"""
Jax default parameter structure:
dict_keys(['Dense_0', 'Dense_1', 'Dense_2', 'Dense_3', 'Dense_4', 'Dense_5', 'Dense_6', 'Dense_7', 'embedding_table'])

Pytorch stateduct structure:
dict_keys(['embedding_chunk_0', 'embedding_chunk_1', 'embedding_chunk_2', 'embedding_chunk_3', 'bot_mlp.0.weight', 'bot_mlp.0.bias', 'bot_mlp.2.weight', 'bot_mlp.2.bias', 'bot_mlp.4.weight', 'bot_mlp.4.bias', 'top_mlp.0.weight', 'top_mlp.0.bias', 'top_mlp.2.weight', 'top_mlp.2.bias', 'top_mlp.4.weight', 'top_mlp.4.bias', 'top_mlp.6.weight', 'top_mlp.6.bias', 'top_mlp.8.weight', 'top_mlp.8.bias'])



The following function converts the PyTorch weights to the Jax format
and assigns them to the Jax model parameters.
The function assumes that the Jax model parameters are already initialized
and that the PyTorch weights are in the correct format.
"""
def use_pytorch_weights(jax_params):
    # --- Embedding Table ---
    embedding_table = np.concatenate([
        numpy_weights[f'embedding_chunk_{i}'] for i in range(4)
    ], axis=0)  # adjust axis depending on chunking direction

    jax_params['embedding_table'] = embedding_table

    # --- Bot MLP: Dense_0 to Dense_2 ---
    for i, j in zip([0, 2, 4], range(3)):
        jax_params[f'Dense_{j}']['kernel'] = numpy_weights[f'bot_mlp.{i}.weight'].T
        jax_params[f'Dense_{j}']['bias'] = numpy_weights[f'bot_mlp.{i}.bias']

    # --- Top MLP: Dense_3 to Dense_7 ---
    for i, j in zip([0, 2, 4, 6, 8], range(3, 8)):
        jax_params[f'Dense_{j}']['kernel'] = numpy_weights[f'top_mlp.{i}.weight'].T
        jax_params[f'Dense_{j}']['bias'] = numpy_weights[f'top_mlp.{i}.bias']
    
    return jax_params
