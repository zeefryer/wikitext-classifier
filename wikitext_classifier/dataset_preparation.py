"""Download and process datasets for downstream use.

Can be run as a script using
`python dataset_preparation.py --config=[path to config file]`
or imported to use individual functions.

"""

import argparse
import functools
import pathlib
import shutil

import polars as pl
import yaml
from langdetect import DetectorFactory  # to make langdetect deterministic
from sklearn.feature_extraction.text import CountVectorizer

import wikitext_classifier.data_utils.transformation as transformation
import wikitext_classifier.data_utils.utils as utils

DEFAULT_CONFIG_FILENAME = "config/data_prep.yaml"
VALID_DATASET_NAMES = ["wikipedia_talk"]


def get_source_data(config, save_location):
    """Download dataset to specified location.

    Args:
        config: dict, must contain 'source_dataset_name' key.
        save_location: str or pathlib.Path, folder to save data in.

    Returns:
        Dict of paths to dataset files.
    """
    dataset_name = config["source_dataset_name"]
    if isinstance(save_location, str):
        save_location = pathlib.Path(save_location)

    if dataset_name == "wikipedia_talk":
        result = utils.get_wikipedia_talk_data(save_location)
        return result
    else:
        raise RuntimeError(
            f"Please specify a valid dataset name; options are {VALID_DATASET_NAMES}"
        )


def prepare_dataset_splits(config, source_data_files):
    """Split dataset into logical pieces (e.g. train/eval/test splits).

    Note that the returned dict can contain more than just the train/eval/test
    splits (e.g. if a separate file of annotations or landmarks is required).

    Args:
        config: dict, must contain 'source_dataset_name'.
        source_data_files: dict of paths to dataset files, e.g. as obtained from
            get_source_data.

    Returns:
        Dict of dataframes, and list of splits.
    """
    dataset_name = config["source_dataset_name"]

    if dataset_name == "wikipedia_talk":
        splits = ["train", "dev", "test"]
        df_comments = pl.read_csv(source_data_files["comments"])
        raw_dataframes = utils.get_dataset_splits(df_comments, splits, "split")
        raw_dataframes["annotations"] = pl.read_csv(
            source_data_files["annotations"])
        del df_comments
        return raw_dataframes, splits
    else:
        raise RuntimeError(
            f"Please specify a valid dataset name; options are {VALID_DATASET_NAMES}"
        )


def clean_data(config, df):
    """Convenience function to call sequence of initial data cleanup functions.

    Args:
        config: dict, specifies how the data cleanup functions should behave.
        df: pl.Dataframe, the dataframe to be processed.

    Returns:
        pl.Dataframe, the processed dataframe.
    """
    if config["max_comment_length"]:
        char_count_col = f"{config['raw_text_col']}_char_count"
        if char_count_col not in df.columns:
            df = transformation.get_text_lengths(df, config["raw_text_col"],
                                                 char_count_col)
        df = df.filter(pl.col(char_count_col) <= config["max_comment_length"])

    if config["drop_nonenglish"]:
        lang_col = f"{config['raw_text_col']}_language"
        if lang_col not in df.columns:
            df = transformation.get_language(df, config["raw_text_col"],
                                             lang_col)
        df = df.filter(pl.col(lang_col) == "en")

    return df


def preprocessing(config, df):
    """Convenience function to call sequence of data preprocessing functions.

    Per-dataset behaviour can be specified using config['source_dataset_name'].

    Args:
        config: dict, specifies how the preprocessing functions should behave.
        df: pl.Dataframe, the dataframe to be processed.

    Returns:
        pl.Dataframe, the processed dataframe.
    """
    proc_col = f"{config['raw_text_col']}_processed"
    df = df.with_columns(pl.col(config["raw_text_col"]).alias(proc_col))

    # Strip tokens first so that subsequent steps don't modify them
    if config["strip_tabs_and_newlines"]:
        pattern = f"{config['newline_token']}|{config['tab_token']}"
        df = df.with_columns(pl.col(proc_col).str.replace_all(pattern, " "))

    # strip_accents must happen before strip_punctuation
    if config["strip_accents"] or config["lowercase"]:
        vect = CountVectorizer(
            strip_accents=config["strip_accents"],
            lowercase=config["lowercase"])
        preproc_fn = vect.build_preprocessor()
        df = df.with_columns(
            pl.col(proc_col).map_elements(preproc_fn, return_dtype=pl.String))

    if config["strip_punctuation"]:
        if isinstance(config["strip_punctuation"], str):
            pattern = config["strip_punctuation"]
        else:
            pattern = r"[^a-zA-Z-_\' ]"
        df = df.with_columns(pl.col(proc_col).str.replace_all(pattern, " "))

    # strip_whitespace should happen last since the other steps introduce spaces
    if config["strip_whitespace"]:
        df = df.with_columns(
            pl.col(proc_col).str.replace_all(r"\s{2,}", " ")).with_columns(
                pl.col(proc_col).str.strip_chars_start(" ")).with_columns(
                    pl.col(proc_col).str.strip_chars_end(" "))

    return df


def process_dataset(config, raw_dataframes, split, save_location):
    """Wrapper function for cleaning and processing dataset splits.

    Per-dataset behaviour can be specified using config['source_dataset_name'].

    Args:
        config: dict, specifies how the preprocessing functions should behave.
        raw_dataframes: dict, containing all required dataframes for processing.
        split: str, the dataset split to process (should be a key in
            raw_dataframes)
        save_location: str or pathlib.Path, the location to save the processed
            dataset.

    Returns:
        pl.Dataframe, the processed dataframe.
    """
    dataset_name = config["source_dataset_name"]
    if dataset_name == "wikipedia_talk":
        df_text = raw_dataframes[split]
        df_annotations = raw_dataframes["annotations"]
        agg_fn = functools.partial(
            transformation.aggregate_annotations,
            group_col=config["id_col"],
            source_col=config["label_col"],
            agg_type=config["annotation_aggregation"],
            thresh=config["aggregation_avg_threshold"],
        )

        print(
            f"Processing {split} split of {dataset_name}; size {len(df_text)}")

        # Clean up and preprocess the text samples
        df_text = clean_data(config, df_text)
        df_text = preprocessing(config, df_text)
        df_text.write_csv(save_location.joinpath(f"{split}_nolabels.csv"))

        # Combine the text and annotations
        df = utils.combine_text_and_annotations(
            df_text, df_annotations, config["id_col"], aggregation_fn=agg_fn)
        df.write_csv(save_location.joinpath(f"{split}_withlabels.csv"))

        print(
            f"Finished processing {split} split of {dataset_name}; size {len(df_text)}"
        )

        return df
    else:
        raise RuntimeError(
            f"Please specify a valid dataset name; options are {VALID_DATASET_NAMES}"
        )


def main(args):
    # Get the config
    config = utils.get_config(args.config, DEFAULT_CONFIG_FILENAME)

    # Makes langdetect determinstic; delete if not desired
    DetectorFactory.seed = 0

    # Set up paths to various directories and create folders as needed
    root_dir = pathlib.Path(config["root_dir"])

    if config["source_data_dir"] is None:
        source_data_dir = root_dir.joinpath("source_data")
    else:
        source_data_dir = pathlib.Path(config["source_data_dir"])

    data_dir = root_dir.joinpath("data")

    for x in [root_dir, data_dir]:
        if not x.exists():
            x.mkdir(parents=True)

    if config["download_files"]:
        if source_data_dir.exists():
            shutil.rmtree(source_data_dir)
        source_data_dir.mkdir()
        source_data_files = get_source_data(config, source_data_dir)
    else:
        source_data_files = [x for x in source_data_dir.glob("*")]

    raw_dataframes, splits = prepare_dataset_splits(config, source_data_files)
    """
    The following block handles a bug in langdetect multiprocessing: sometimes the
    lang profiles aren't loaded in time and langdetect throws a LangDetectException.
    Unfortunately it also throws a LangDetectException if it fails on a 
    single instance (e.g. text is just a url) and there's easy no way to disambiguate.
    This is solved by calling a small "dummy" run of langdetect before the main loop.
    """
    if config["drop_nonenglish"]:
        _ = transformation.get_language(raw_dataframes[splits[0]].head(20),
                                        config["raw_text_col"])

    # Now call the preprocessing steps on each dataframe.
    dataframes = {}
    for split in splits:
        dataframes[split] = process_dataset(config, raw_dataframes, split,
                                            data_dir)

    return dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()
    main(args)
