import wikitext_classifier.models as models
import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import numpy as np


"""
inputs: config, weights path, string or csv to inference on
outputs: printout? where to save csv

call pad_batch on all batches, right before calling inference_step
decide whether to jit based on whether we have >1 batch

todo: are we going to use the dataset/dataloader functionality at all?
if not: how to handle things here? just use the collate function? 
collate fn assumes presence of labels
"""


def inference_step(state, batch):
    logits = state.apply_fn({"params": state.params},
                        batch["inputs"],
                        train=False)
    
    # throw away any padding
    logits = logits[:batch["bs"],:]

    probs = nn.softmax(logits)
    preds = jnp.argmax(probs, axis=-1)
    return preds, probs


def batch_single_input(input_string):
    # figure out the correct tokenizer settings (turn truncation off)
    # then figure out if the current defaults are correct and fix if not
    pass


def create_batch():
    # might not need this function, this might be a while loop in the main fn
    pass


def pad_batch(batch, target_batch_size, batch_axis=0):
    # note that for now we don't change the shape of batch['text'] or batch['label']
    # put in docstring: reason for padding
    # this might want to go live elsewhere? since it'll be used in the eval loop too
    if batch.shape[batch_axis] == target_batch_size:
        return batch
    dim0, dim1 = batch['inputs']['input_ids'].shape
    padding = np.zeros((target_batch_size - dim0, dim1), dtype=int)
    new_input_ids = np.concat(batch['inputs']['input_ids'], padding)
    new_attention_mask = np.concat(batch['inputs']['attention_mask'], padding)
    batch['inputs'] = {'input_ids': new_input_ids, 'attention_mask': new_attention_mask}
    return batch
    



def main():
    clf = models.roberta_load_model(config, weights_path)
    state = TrainState() # todo: this is not how you create a trainstate
