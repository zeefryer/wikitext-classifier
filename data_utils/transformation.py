"""Collection of functions to transform datasets."""

import polars as pl
import langdetect

from langdetect.lang_detect_exception import LangDetectException


def detect_language(s):
    """Uses langdetect to predict the language of a single string s."""
    try:
        lang = langdetect.detect(s)
    except LangDetectException:
        lang = None
    return lang


def get_text_lengths(df, source_col, target_col=None):
    """Create dataframe column with length (in characters) of each item."""
    if target_col is None:
        target_col = f"{source_col}_char_count"
    return df.with_columns(pl.col(source_col).str.len_chars().alias(target_col))


def get_language(df, source_col, target_col=None):
    """Create dataframe column with predicted language of each item."""
    if target_col is None:
        target_col = f"{source_col}_language"
    return df.with_columns(
        pl.col(source_col)
        .map_elements(detect_language, return_dtype=pl.String)
        .alias(target_col)
    )


def aggregate_annotations(
    df, group_col, source_col, agg_type="majority", thresh=0.5
):
    """Combine multiple annotations per item into one consolidated label.

    Can specify new aggregation strategies by defining new 'agg_type' rules.

    Caution: the returned dataframe will not be the same size as the input
    dataframe!

    Args:
        df: pl.Dataframe, the dataframe of annotations.
        group_col: str, the column name to group by.
        source_col: str, the name of the column containing the annotations that
            are to be aggregated.
        agg_type: str, specify the aggregation strategy.
        thresh: float, default 0.5. Only used in the "average" aggregation 
            strategy, to specify the positive/negative cutoff point.

    Returns:
        pl.Dataframe with the aggregation strategy applied.
    """
    # be careful to return a subset of columns here due to the group_by
    if agg_type == "majority":
        return (
            df.group_by(pl.col(group_col))
            .agg(pl.col(source_col).mode())
            .with_columns(pl.col(source_col).list.min().cast(pl.Int8))
            .select(pl.col([group_col, source_col]))
        )
    elif agg_type == "average":
        return (
            df.group_by(pl.col(group_col))
            .agg(pl.col(source_col).mean())
            .with_columns((pl.col(source_col) > thresh).cast(pl.Int8))
            .select(pl.col([group_col, source_col]))
        )
    else:
        raise RuntimeError
