import wikitext_classifier.models as models
import wikitext_classifier.model_utils as model_utils
import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import numpy as np
import pathlib
import argparse
import optax
import yaml
import functools

import wikitext_classifier.data_utils.utils as utils


"""
inputs: config, weights path, string or csv to inference on
outputs: printout? where to save csv

call pad_batch on all batches, right before calling inference_step
decide whether to jit based on whether we have >1 batch

todo: are we going to use the dataset/dataloader functionality at all?
if not: how to handle things here? just use the collate function? 
collate fn assumes presence of labels
"""

DEFAULT_CONFIG_FILENAME = "config/inference.yaml"


def inference_step(state, batch):
    logits = state.apply_fn({"params": state.params},
                        batch["inputs"],
                        train=False)
    
    # throw away any padding
    logits = logits[:batch["bs"],:]

    probs = nn.softmax(logits)
    preds = jnp.argmax(probs, axis=-1)
    return preds, probs


def estimate_num_batches():
    pass


def batch_single_input(input_string, tokenizer):
    


def create_batch():
    # might not need this function, this might be a while loop in the main fn
    pass


def main(args):
    # Get the config
    config = utils.get_config(args.config, DEFAULT_CONFIG_FILENAME)

    trained_weights_path = config['weights_path']
    clf = models.roberta_load_model(config, trained_weights_path)

    # Tokenizer should NOT truncate sequences, we will handle this manually.
    clf["tokenizer"] = functools.partial(clf["tokenizer"], truncation=False)

    # Create a "train" state for convenience
    tx = optax.sgd(1.0)  # not used; only required to init state below
    state = TrainState.create(
        apply_fn=clf['model'].apply, params=clf['variables']['params'], tx=tx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--input", help="String or csv to run inference on")
    args = parser.parse_args()
    main(args)
