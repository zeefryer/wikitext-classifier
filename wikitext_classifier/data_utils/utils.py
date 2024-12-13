"""Util functions related to dataset preparation.
"""
import pathlib

import polars as pl
import requests

from data_utils.dataset import TextDataset


def consolidate_data(files):
    data = [pl.read_csv(x) for x in files]
    return pl.concat(data)


def get_dataset_splits(df, splits, source_col):
    return {x: df.filter(pl.col(source_col) == x) for x in splits}


def combine_text_and_annotations(df_text,
                                 df_annotations,
                                 join_col,
                                 aggregation_fn=None):
    """Combine text and annotation dataframes, if initially separate.

    Args:
        df_text: pl.Dataframe containing text samples.
        df_annotations: pl.Dataframe containing annotations/labels.
        join_col: str, the column name on which to join the two dataframes.
        aggregation_fn: optional. If provided, is called on df_annotation before
            joining the two frames (e.g. to handle the case where there are
            multiple labels per sample).

    Returns:
        pl.Dataframe, the combined dataframe.
    """
    if aggregation_fn is not None:
        df_ann = aggregation_fn(df_annotations)
    else:
        df_ann = df_annotations
    return df_text.join(df_ann, on=join_col, how="left")


def df_col_to_list(df, source_col):
    """Convert a single polars dataframe column to a list."""
    return df.select(pl.col(source_col)).to_series().to_list()


def df_to_dataset(df, data_col, label_col):
    """Convert a polars dataframe to a TextDataset instance.

    Args:
        df: pl.Dataframe, the source dataframe.
        data_col: str, the column name for the samples.
        label_col: str, the column name for the labels.

    Returns:
        TextDataset
    """
    return TextDataset(
        df_col_to_list(df, data_col), df_col_to_list(df, label_col))


def get_wikipedia_talk_data(save_location):
    """Convenience function to download the Wikipedia Talk dataset.

    See https://github.com/ewulczyn/wiki-detox/tree/master

    Args:
        save_location: str or pathlib.Path, where to save the dataset to.

    Returns:
        dict, with paths to the dataset's 'comments' and 'annotations' csvs.
    """
    urls = {
        "comments": "https://ndownloader.figshare.com/files/7554634",
        "annotations": "https://ndownloader.figshare.com/files/7554637",
    }
    query_parameters = {"downloadformat": "tsv"}

    if isinstance(save_location, str):
        save_location = pathlib.Path(save_location)

    for k, url in urls.items():
        response = requests.get(url, params=query_parameters)
        with open(save_location.joinpath(f"{k}.tsv"), "wb") as f:
            f.write(response.content)

    result = {x.stem: x for x in save_location.iterdir()}

    # Cleanup: convert from tsv to csv
    for k in result:
        v = result[k]
        pl.read_csv(
            v, separator="\t").write_csv(
                v.with_suffix(".csv"), separator=",")
        result[k] = v.with_suffix(".csv")

    return result
