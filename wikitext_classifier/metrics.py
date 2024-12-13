import jax
import jax.numpy as jnp
import optax


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
