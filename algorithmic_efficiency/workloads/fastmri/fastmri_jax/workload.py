"""FastMRI workload implemented in Jax."""

import functools
import math
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.fastmri.fastmri_jax import models
from algorithmic_efficiency.workloads.fastmri.ssim import ssim
from algorithmic_efficiency.workloads.fastmri.workload import \
    BaseFastMRIWorkload


class FastMRIWorkload(BaseFastMRIWorkload):

  @property
  def model_params_types(self):
    """The shapes of the parameters in the workload model."""
    if self._param_types is None:
      self._param_types = param_utils.jax_param_types(self._param_shapes)
    return self._param_types

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    fake_batch = jnp.zeros((13, 320, 320))
    variables = jax.jit(models.UNet().init)({'params': rng}, fake_batch)
    params = variables['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    params = jax_utils.replicate(params)
    return params, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      dropout_rate: Optional[float],
      aux_dropout_rate: Optional[float],
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """aux_dropout_rate is unused."""
    del model_state
    del aux_dropout_rate
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits = models.UNet(dropout_rate=dropout_rate).apply(
        {'params': params},
        augmented_and_preprocessed_input_batch['inputs'],
        rngs={'dropout': rng},
        train=train)
    return logits, None

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:

    activation_fn = {
        spec.LossType.SOFTMAX_CROSS_ENTROPY: jax.nn.softmax,
        spec.LossType.SIGMOID_CROSS_ENTROPY: jax.nn.sigmoid,
        spec.LossType.MEAN_SQUARED_ERROR: lambda z: z,
        spec.LossType.MEAN_ABSOLUTE_ERROR: lambda z: z
    }
    return activation_fn[loss_type](logits_batch)

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor,
              outputs_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    return jnp.abs(outputs_batch -
                   label_batch).mean(axis=tuple(range(1, outputs_batch.ndim)))

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0),
      static_broadcasted_argnums=(0,))
  def _eval_model(self, params, batch, rng):
    """Return the SSIM and loss as a dict."""
    outputs, _ = self.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.EVAL,
        rng=rng,
        dropout_rate=0.0,  # Not relevant for eval.
        aux_dropout_rate=None,
        update_batch_norm=False)
    ssim_vals = ssim(
        batch['targets'],
        outputs,
        mean=batch['mean'],
        std=batch['std'],
        volume_max=batch['volume_max'])
    ssim_sum = jnp.sum(ssim_vals)
    loss = jnp.sum(self.loss_fn(batch['targets'], outputs))
    metrics = {
        'ssim': ssim_sum,
        'loss': loss,
        'weight': jnp.sum(batch['weights']),
    }
    metrics = jax.lax.psum(metrics, axis_name='batch')
    return metrics

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0):
    """Run a full evaluation of the model."""
    del model_state
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self.build_input_queue(
          data_rng,
          split,
          data_dir,
          global_batch_size=global_batch_size,
          repeat_final_dataset=True)

    total_metrics = {'ssim': 0., 'loss': 0., 'weight': 0.}
    num_batches = int(math.ceil(num_examples / global_batch_size))
    eval_rngs = prng.split(model_rng, jax.local_device_count())
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      # We already sum these metrics across devices inside _compute_metrics.
      synced_metrics = self._eval_model(params, batch, eval_rngs)
      total_metrics = {
          k: v + synced_metrics[k][0] for k, v in total_metrics.items()
      }
    num_examples = total_metrics.pop('weight')
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}