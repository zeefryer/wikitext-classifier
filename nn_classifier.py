"""JAX/Flax neural net classifier on Roberta backbone.

If called as as script, e.g. 
`python nn_classifier.py --config=[path to config]`
will train a simple dense text classifier on top of a RoBERTa backbone.

However any function that does not reference RoBERTa in its name is intended to
be model-agnostic, so an alternative backbone and/or classifier can easily be
swapped in with minimal changes.
"""

from collections.abc import Callable

import argparse
import datetime
import functools
import json
import pathlib
import yaml

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl

from flax.training import train_state, checkpoints

from transformers import AutoTokenizer, FlaxRobertaModel
from torch.utils.data import DataLoader

from data_utils.dataset import TextDataset
from data_utils.utils import df_to_dataset
"""
TODO
learning rate scheduler?
pad final batch size
"""

DEFAULT_HF_ROBERTA = "FacebookAI/roberta-base"
DEFAULT_CONFIG_FILENAME = "train_config.yaml"


class TrainState(train_state.TrainState):
  dropout_key: jax.Array
  pos_weight: jnp.float32 = 1.0
  neg_weight: jnp.float32 = 1.0


class RobertaTextClassifier(nn.Module):
  backbone: nn.Module
  features: tuple[int]

  def setup(self):
    self.head = [nn.Dense(feat) for feat in self.features]

  def __call__(self, x, train: bool):
    x = self.backbone(**x, deterministic=not train)["last_hidden_state"]
    x = x[:, 0, :]  # take the output corresponding to [CLS]
    for i, L in enumerate(self.head):
      x = L(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x


def create_roberta_classifier_from_hf(config):
  """Initialize a pretrained RoBERTa classifier from huggingface weights.

    Given the name of any pretrained hugginface RoBERTa model, this function
    downloads the corresponding weights, model definition, and tokenizer, and 
    creates a new RobertaTextClassifier of type nn.Module consisting of the
    pretrained RoBERTa backbone and a randomly-initialized classifier head.

    Args:
        config: dict. Must contain 'params_key' (a jax.random.PRNGKey), 
            'classifier_head_dims' (a tuple specifying the number and size of
            classifier head layers), and 'backbone_hf_str' (the huggingface
            string specifying the RoBERTa model).

    Returns:
        Dict containing the model definition, pytree of model variables, and
            tokenizer.
    """
  rng = config["params_key"]
  backbone_hf_str = config["backbone_hf_str"] or DEFAULT_HF_ROBERTA

  if config["classifier_head_dims"] is None:
    classifier_head_dims = (2,)
  else:
    classifier_head_dims = tuple(config["classifier_head_dims"])

  # dummy sentence for initializing the models
  test_sentence = "Here is a random sentence about cats."

  # tokenizer setup
  tokenizer = AutoTokenizer.from_pretrained(backbone_hf_str)
  tokenizer = functools.partial(
      tokenizer, padding="max_length", truncation=True, return_tensors="jax")

  # get the pretrained Roberta model and extract its module and variables
  backbone = FlaxRobertaModel.from_pretrained(
      backbone_hf_str, add_pooling_layer=False)
  backbone_module = backbone.module
  backbone_params = backbone.params

  # create and initialize the classifier
  clf = RobertaTextClassifier(backbone_module, classifier_head_dims)
  inputs = tokenizer(test_sentence)
  variables = clf.init(rng, inputs, train=False)

  # update the classifier variables to include the pretrained backbone variables
  variables["params"]["backbone"] = backbone_params

  return {"model": clf, "variables": variables, "tokenizer": tokenizer}


def roberta_collate_fn(data, tokenizer):
  """RoBERTA-specific collate function for torch dataloader."""
  text, labels = zip(*data)
  inputs = dict(tokenizer(list(text)))
  labels = np.array(labels)
  return {"text": text, "inputs": inputs, "label": labels}


def get_frozen_param_partition(params, rule="backbone"):
  """Specify frozen/trainable parameters for optax.multi_transform.
    """
  if rule == "backbone":
    partition = flax.traverse_util.path_aware_map(
        lambda p, v: "frozen" if "backbone" in p else "trainable", params)
  else:
    # default option is to make everything trainable
    partition = flax.traverse_util.path_aware_map(lambda p, v: "trainable",
                                                  params)
  return partition


def get_tx(params, train_tx, rule='backbone'):
  """Create an optimizer that freezes a subset of the parameters.

    Args:
        params: JAX pytree of model parameters.
        train_tx: optax optimizer for training the non-frozen parameters.
        rule: str, is passed to get_frozen_param_partition to specify which
            parameters are frozen.

    Returns:
        optax optimizer
    """
  partition_optimizers = {
      "trainable": train_tx,
      "frozen": optax.set_to_zero(),
  }
  param_partitions = get_frozen_param_partition(params, rule)
  tx = optax.multi_transform(partition_optimizers, param_partitions)
  return tx


def compute_weighted_accuracy(logits, labels, weights=None):
  """Compute weighted accuracy on a batch.

    Args:
        logits: jax array of logits.
        labels: jax array of true labels (same shape as logits).
        weights: optional, weights to be applied to each item (same shape as
            logits). If not specified, all items are given weight 1.

    Returns:
        tuple: (weighted accuracy sum, denominator). Actual accuracy on the
            batch can be obtained as (weighted accuracy sum)/(denominator).
    """
  acc = jnp.equal(jnp.argmax(logits, axis=-1), labels)
  denom = jnp.array(len(labels)).astype(jnp.float32)
  if weights is not None:
    acc = acc * weights
    denom = weights.sum()
  return acc.sum(), denom


def compute_weighted_cross_entropy(logits, labels, weights=None):
  """Compute weighted cross entropy on a batch.

    Args:
        logits: jax array of logits.
        labels: jax array of true labels (same shape as logits).
        weights: optional, weights to be applied to each item (same shape as
            logits). If not specified, all items are given weight 1.

    Returns:
        tuple: (weighted loss sum, denominator). Actual loss on the
            batch can be obtained as (weighted loss sum)/(denominator).
    """
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
  denom = jnp.array(len(labels)).astype(jnp.float32)
  if weights is not None:
    loss = loss * weights
    denom = weights.sum()
  return loss.sum(), denom


def compute_metrics(state, logits, labels, weights=None):
  """Compute desired metrics on a given batch.

    Note: convention for keys is that anything prefixed with w_ is weighted
    and should be divided by w_denom to obtain the mean; all others are 
    presumed unweighted should be divided by the batch size.

    Args:
        state: TrainState, the current state of the model.
        labels: jax array of true labels (same shape as logits).
        weights: optional, weights to be applied to each item (same shape as
            logits). If not specified, all items are given weight 1.

    Returns:
        dictionary of metrics.
    """
  metrics = {}

  loss, w_denom = compute_weighted_cross_entropy(logits, labels, weights)
  acc, _ = compute_weighted_accuracy(logits, labels, weights=None)
  w_acc, _ = compute_weighted_accuracy(logits, labels, weights)

  metrics["w_loss"] = loss
  metrics["acc"] = acc
  metrics["w_acc"] = w_acc
  metrics["denom"] = jnp.array(len(labels))
  metrics["w_denom"] = w_denom

  return metrics


def consolidate_metrics(metrics):
  """Combines metrics from all batches and returns overall score.

    Args:
        metrics: iterable of dicts, each dict the output of compute_metrics.

    Returns:
        dict of consolidated metrics.
    """
  metrics_t = jax.tree.transpose(
      jax.tree.structure([0 for e in metrics]),
      jax.tree.structure(metrics[0]),
      metrics,
  )
  metrics_sum = {k: jnp.array(v).sum().item() for (k, v) in metrics_t.items()}
  w_denom = metrics_sum.pop("w_denom")
  denom = metrics_sum.pop('denom')

  metrics_denoms = {
      k: w_denom if k.startswith('w_') else denom for k in metrics_sum
  }

  metrics_mean = jax.tree_util.tree_map(lambda x, y: x / y, metrics_sum,
                                        metrics_denoms)

  metrics_mean['batch_size'] = denom
  metrics_mean['total_weight'] = w_denom

  return metrics_mean


@jax.jit
def train_step(state, batch):
  dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)
  weights = jnp.where(batch["label"] > 0, state.pos_weight,
                      state.neg_weight).astype(jnp.float32)

  def loss_fn(params):
    logits = state.apply_fn(
        {"params": params},
        batch["inputs"],
        train=True,
        rngs={"dropout": dropout_train_key},
    )

    loss, denom = compute_weighted_cross_entropy(logits, batch["label"],
                                                 weights)
    mean_loss = loss / denom

    return mean_loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(state, logits, batch["label"], weights)

  return new_state, metrics


@jax.jit
def eval_step(state, batch):
  weights = jnp.where(batch["label"] > 0, state.pos_weight,
                      state.neg_weight).astype(jnp.float32)

  logits = state.apply_fn({"params": state.params},
                          batch["inputs"],
                          train=False)
  metrics = compute_metrics(state, logits, batch["label"], weights)

  return metrics


def evaluate(state, dl_eval):
  """Runs an evaluation loop on the provided dataset.

    Args:
        state: TrainState, the current model state.
        dl_eval: torch.DataLoader, the dataset to evaluate on.

    Returns:
        dict of metrics computed on the dataset.
    """
  iter_eval = iter(dl_eval)
  eval_metrics = []
  for i, batch in enumerate(iter_eval):
    metrics = eval_step(state, {
        "inputs": batch["inputs"],
        "label": batch["label"]
    })
    eval_metrics.append(metrics)
    print(f"Eval progress: {i+1}/{len(iter_eval)}", end='\r')
  return consolidate_metrics(eval_metrics)


def prepare_data(config):
  """Prepares the data for training.

    Specify input data and splits in the config file using 'dataset_paths'.

    Args:
        config: dict

    Returns:
        dict of TextDataset instances
    """
  datasets = {k: pl.read_csv(x) for (k, x) in config["dataset_paths"].items()}
  result = {}

  text_col = config["dataset_text_column"]
  label_col = config["dataset_label_column"]

  if "train" in datasets:
    if config["train_sample_max_length"] is not None:
      df_train = datasets["train"].filter(
          pl.col("comment_char_count") <= config["train_sample_max_length"])
    else:
      df_train = datasets["train"]
    result["train"] = df_to_dataset(df_train, text_col, label_col)

  if "eval" in datasets:
    df_eval = datasets["eval"]
    result["eval"] = df_to_dataset(df_eval, text_col, label_col)

  if "test" in datasets:
    df_test = datasets["test"]
    result["test"] = df_to_dataset(df_test, text_col, label_col)

  return result


def train(config, clf, datasets):
  """Main training loop.

    Args:
        config: dict with training hyperparameters.
        clf: dict, containing (at least) model', 'variables', and 'tokenizer'.
        datasets: dict, containing (at least) 'train' and (if config['do_eval'])
            is True) must contain 'eval' as well.

    Returns:
        pathlib.Path, the location of the final checkpoint.
    """
  train_tx = optax.adamw(config["lr"])
  tx = get_tx(clf["variables"]["params"], train_tx)

  if config["pos_weight"] is None:
    config["pos_weight"] = 1.0
  if config["neg_weight"] is None:
    config["neg_weight"] = 1.0

  state = TrainState.create(
      apply_fn=clf["model"].apply,
      params=clf["variables"]["params"],
      tx=tx,
      dropout_key=config["dropout_key"],
      pos_weight=config["pos_weight"],
      neg_weight=config["neg_weight"],
  )

  train_metrics = []

  dl_train = DataLoader(
      datasets["train"],
      batch_size=config["train_batch_size"],
      shuffle=True,
      collate_fn=functools.partial(
          roberta_collate_fn, tokenizer=clf["tokenizer"]),
  )
  if config["do_eval"]:
    dl_eval = DataLoader(
        datasets["eval"],
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=functools.partial(
            roberta_collate_fn, tokenizer=clf["tokenizer"]),
    )

  num_steps_to_train = config["num_steps_to_train"]
  num_steps_per_eval = config["num_steps_per_eval"]
  checkpoint_freq = config["checkpoint_freq"]

  iter_train = iter(dl_train)
  while state.step < num_steps_to_train:
    print(f"Train step: {state.step+1}/{num_steps_to_train}", end="\r")
    try:
      batch = next(iter_train)
    except StopIteration:
      iter_train = iter(dl_train)
      batch = next(iter_train)

    state, metrics = train_step(state, {
        "inputs": batch["inputs"],
        "label": batch["label"]
    })
    metrics = {
        k: v.item() for (k, v) in metrics.items()
    }  # can't serialize ArrayImpl type
    train_metrics.append(metrics)
    is_last_step = state.step == num_steps_to_train

    if config["do_eval"]:
      if num_steps_per_eval is None:
        num_steps_per_eval = num_steps_to_train
      if state.step > 0 and (state.step % num_steps_per_eval == 0 or
                             is_last_step):
        print(f"Running eval loop at step {state.step}")
        eval_metrics = evaluate(state, dl_eval)
        print(f"Eval metrics at step {state.step}: {eval_metrics}")
        with open(config["metrics_dir"].joinpath("eval.jsonl"), "a") as f:
          json.dump(eval_metrics, f)
          f.write("\n")

    if state.step % 100 == 0 or is_last_step:
      with open(config["metrics_dir"].joinpath("train.jsonl"), "a") as f:
        json.dump(train_metrics, f)
        f.write("\n")

    if state.step % checkpoint_freq == 0 or is_last_step:
      checkpoints.save_checkpoint(
          config["checkpoints_dir"], target=state, step=state.step)

  final_checkpoint = checkpoints.latest_checkpoint(config["checkpoints_dir"])
  return final_checkpoint


def main(args):
  # Get the config
  if args.config is None:
    cwd = pathlib.Path(__file__).parent
    config_path = cwd.joinpath(DEFAULT_CONFIG_FILENAME)
  else:
    config_path = pathlib.Path(args.config)

  with open(config_path) as f:
    train_config = yaml.safe_load(f)

  print(train_config)

  # Use the current date/time to disambiguate training runs
  now = datetime.datetime.now()
  now = datetime.datetime.strftime(now, "%Y%m%d_%H%M%S")

  # Initialize our RNG keys; re-split 'key' if another key is needed in future.
  key = jax.random.PRNGKey(train_config["random_seed"])
  key, params_key, dropout_key = jax.random.split(key, 3)

  # A bit verbose but helps with debugging if we're explicit about keys here
  train_config["key"] = key
  train_config["params_key"] = params_key
  train_config["dropout_key"] = dropout_key

  # Set up subdirs for saving metrics and checkpoints for this run
  if train_config["save_dir"] is None:
    train_config["save_dir"] = pathlib.Path(
        train_config["root_dir"]).joinpath("results")
  else:
    train_config["save_dir"] = pathlib.Path(train_config["save_dir"])
  for x in ["metrics", "checkpoints"]:
    p = train_config["save_dir"].joinpath(now, x)
    p.mkdir(parents=True, exist_ok=False)
    train_config[f"{x}_dir"] = p
  print(f"Saving results to {train_config['metrics_dir'].parent}")

  # Save a copy of the config for reference
  with open(train_config["metrics_dir"].parent.joinpath("train_config.yaml"),
            "w") as f:
    yaml.dump(train_config, f)

  # Set up the datasets and models
  datasets = prepare_data(train_config)
  clf = create_roberta_classifier_from_hf(train_config)

  # Finally, trigger the training loop
  trained_model_dir = train(train_config, clf, datasets)
  print(f"Location of final checkpoint: {trained_model_dir}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", help="Path to config file")
  args = parser.parse_args()
  main(args)
