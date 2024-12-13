import functools

import flax.linen as nn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, FlaxRobertaModel

DEFAULT_HF_ROBERTA = "FacebookAI/roberta-base"


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


def create_roberta_classifier_from_hf(config, pretrained=True):
    """Initialize a RoBERTa classifier from huggingface.

    Given the name of any huggingface RoBERTa model, this function downloads the
    model definition, tokenizer, and (optionally) pretrained weights, and
    creates a new RobertaTextClassifier of type nn.Module consisting of the
    RoBERTa backbone and a randomly-initialized classifier head.

    Use `pretrained=False` to speed up model initialization when providing your
    own weights (e.g. from a completed training run). In this setting the config
    entries `classifier_head_dims` and `backbone_hf_str` need to match those
    from your training run, but `params_key` can be any arbitrary
    jax.random.PRNGKey (since it is only used to initialize parameters that are
    later overwritten).

    Args:
        config: dict. Must contain 'params_key' (a jax.random.PRNGKey), 
            'classifier_head_dims' (a tuple specifying the number and size of
            classifier head layers), and 'backbone_hf_str' (the huggingface
            string specifying the RoBERTa model).
        pretrained: bool, default True. If True, downloads the pretrained model
            weights too; if not, just returns a randomly initialized model.

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

    if pretrained:
        # get the pretrained Roberta model and extract its module and variables
        backbone = FlaxRobertaModel.from_pretrained(
            backbone_hf_str, add_pooling_layer=False)
        backbone_module = backbone.module  # pyright: ignore
        backbone_params = backbone.params  # pyright: ignore
    else:
        r_config = AutoConfig.from_pretrained(config["backbone_hf_str"])
        backbone = FlaxRobertaModel(r_config, add_pooling_layer=False)
        backbone_module = backbone.module

    # create and initialize the classifier
    clf = RobertaTextClassifier(backbone_module, classifier_head_dims)
    inputs = tokenizer(test_sentence)
    variables = clf.init(rng, inputs, train=False)

    if pretrained:
    # update the classifier variables to include the pretrained backbone variables
        variables["params"]["backbone"] = backbone_params  # pyright: ignore

    return {"model": clf, "variables": variables, "tokenizer": tokenizer}


def roberta_collate_fn(data, tokenizer):
    """RoBERTA-specific collate function for torch dataloader."""
    text, labels = zip(*data)
    inputs = dict(tokenizer(list(text)))
    labels = np.array(labels)
    return {"text": text, "inputs": inputs, "label": labels}
