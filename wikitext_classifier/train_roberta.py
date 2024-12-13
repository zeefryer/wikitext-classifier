"""JAX/Flax neural net classifier on Roberta backbone.

If called as as script, e.g. 
`python train_roberta.py --config=[path to config]`
will train a simple dense text classifier on top of a RoBERTa backbone.

The overall train/eval loop is model-agnostic, so in order to swap in a
different backbone or classifier:
    - define the model in models.py, following the RoBERTa example.
    - update which model is loaded in the main function, and (if necessary) the
        dataset collation function in the train function.
"""

# pyright: reportInvalidTypeForm=false
# pyright: reportPrivateImportUsage=false

import argparse
import datetime
import functools
import json
import pathlib

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
import yaml
from flax.training import checkpoints, train_state
from torch.utils.data import DataLoader

import wikitext_classifier.models as models
import wikitext_classifier.metrics as metrics
from wikitext_classifier.data_utils.utils import df_to_dataset
"""
TODO
learning rate scheduler?
pad final batch size
"""

DEFAULT_CONFIG_FILENAME = "config/train.yaml"


class TrainState(train_state.TrainState):
    dropout_key: jax.Array
    pos_weight: jnp.float32 = 1.0
    neg_weight: jnp.float32 = 1.0


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
    tx = optax.multi_transform(partition_optimizers, # pyright: ignore
                               param_partitions)  
    return tx


@jax.jit
def train_step(state, batch):
    dropout_train_key = jax.random.fold_in(
        key=state.dropout_key, data=state.step)
    weights = jnp.where(batch["label"] > 0, state.pos_weight,
                        state.neg_weight).astype(jnp.float32)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch["inputs"],
            train=True,
            rngs={"dropout": dropout_train_key},
        )

        loss, denom = metrics.compute_weighted_cross_entropy(
            logits, batch["label"], weights)
        mean_loss = loss / denom

        return mean_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    metrics = metrics.compute_metrics(state, logits, batch["label"], weights)

    return new_state, metrics


@jax.jit
def eval_step(state, batch):
    weights = jnp.where(batch["label"] > 0, state.pos_weight,
                        state.neg_weight).astype(jnp.float32)

    logits = state.apply_fn({"params": state.params},
                            batch["inputs"],
                            train=False)
    metrics = metrics.compute_metrics(state, logits, batch["label"], weights)

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
    return metrics.consolidate_metrics(eval_metrics)


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
                pl.col("comment_char_count") <=
                config["train_sample_max_length"])
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
            models.roberta_collate_fn, tokenizer=clf["tokenizer"]),
    )
    if config["do_eval"]:
        dl_eval = DataLoader(
            datasets["eval"],
            batch_size=config["eval_batch_size"],
            shuffle=False,
            collate_fn=functools.partial(
                models.roberta_collate_fn, tokenizer=clf["tokenizer"]),
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
                eval_metrics = evaluate(state, dl_eval)  # pyright: ignore
                print(f"Eval metrics at step {state.step}: {eval_metrics}")
                with open(config["metrics_dir"].joinpath("eval.jsonl"),
                          "a") as f:
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
    clf = models.create_roberta_classifier_from_hf(
        train_config, pretrained=True)

    # Finally, trigger the training loop
    trained_model_dir = train(train_config, clf, datasets)
    print(f"Location of final checkpoint: {trained_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()
    main(args)
