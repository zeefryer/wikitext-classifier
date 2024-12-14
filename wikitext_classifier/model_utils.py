import pathlib
from flax.serialization import msgpack_serialize, msgpack_restore
from flax.training.checkpoints import restore_checkpoint
import numpy as np

def load_model_variables(path):
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.is_file() and path.suffix == '.msgpack':
        # flax serialization format
        with open(path, 'rb') as f:
            tmp = f.read()
            result = msgpack_restore(tmp)
            if 'params' in result.keys():  # pyright: ignore
                return result
            elif 'backbone' in result.keys():  # pyright: ignore
                return {'params': result}
    elif path.is_dir():
        if "_CHECKPOINT_METADATA" in [x.name for x in path.iterdir()]:
            # then we have a native flax checkpoint
            variables = extract_params_from_checkpoint(path)
            return variables

    raise RuntimeError(f'Failed to load model params. You provided path {path}')


def extract_params_from_checkpoint(path_to_checkpoint):
    state_dict = restore_checkpoint(path_to_checkpoint, target=None)
    params = state_dict.pop('params')
    return {'params': params}


def pad_batch(batch, target_batch_size, batch_axis=0):
    # note that for now we don't change the shape of batch['text'] or batch['label']
    # put in docstring: reason for padding
    # this might want to go live elsewhere? since it'll be used in the eval loop too
    if batch.shape[batch_axis] == target_batch_size:
        return batch
    dim0, dim1 = batch['inputs']['input_ids'].shape
    padding = np.zeros((target_batch_size - dim0, dim1), dtype=int)
    new_input_ids = np.concatenate(batch['inputs']['input_ids'], padding)
    new_attention_mask = np.concatenate(batch['inputs']['attention_mask'], padding)
    batch['inputs'] = {'input_ids': new_input_ids, 'attention_mask': new_attention_mask}
    return batch