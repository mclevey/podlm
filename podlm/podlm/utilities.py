import re
import yaml
import pandas as pd
import logging


def load_configs(
    config_path: str = "../input/config.yaml",
    private_path: str = "../../../private.yaml",
):
    with open(config_path, "rb") as file:
        config = yaml.safe_load(file)
    with open(private_path, "rb") as file:
        private = yaml.safe_load(file)
    return config, private


def configure_logging(task_name: str):
    logfile = task_name.split("/")[-1].replace(".py", ".log")
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s:%(message)s",
    )


def load_parquet(filename: str):
    return pd.read_parquet(f"../input/{filename}.parquet.gzip")


def save_parquet(df: pd.DataFrame, filename: str):
    df.to_parquet(f"../output/{filename}.parquet.gzip", compression="gzip")


def merge_string_columns(
    df: pd.DataFrame, col1: pd.Series, col2: pd.Series, new_col_name: str, drop=True
):
    df[col1].replace("[deleted]", "", inplace=True)
    df[col2].replace("[deleted]", "", inplace=True)
    df[new_col_name] = df[col1].fillna("") + df[col2].fillna("")
    if drop is True:
        df = df.drop(columns=[col1, col2])
    return df


def split_id_sentence_strings(s: str) -> str:
    pattern = r"[^_]*_[^_]*"
    match = re.search(pattern, s)
    if match:
        id_post = match.group()
        sentence_position_in_post = s.split("_")[2]
        return id_post, sentence_position_in_post
    else:
        return None


def split_ids(df: pd.DataFrame, id_sentence: str = "id_sentence") -> pd.DataFrame:
    df[["post_id", "sentence_position_in_post"]] = (
        df[id_sentence].apply(split_id_sentence_strings).apply(pd.Series)
    )
    return df


def reshape_sentiment_df(
    df: pd.DataFrame,
    id_vars: str = "id",
    value_vars=["P(negative)", "P(neutral)", "P(positive)"],
):
    df_long = pd.melt(df, id_vars, value_vars)
    df_long.columns = ["id", "sentiment", "probability"]
    return df_long


def merge_feature_dfs(df_text_tasks, df_as_collected, on_cols):
    merged = pd.merge(df_text_tasks, df_as_collected, on=on_cols)
    return merged


def group_and_aggregate_author_level(
    df, group_cols, sum_cols, str_cols, join_str_with=" "
):
    grouped = df.groupby(group_cols, as_index=False)
    summed = grouped[sum_cols].sum()
    for col in str_cols:
        concat_concat = grouped[col].agg(lambda x: join_str_with.join(x))
        summed[col + "_concat"] = concat_concat[col]
    return summed


def get_largest_series(s1, s2, s3):
    df = pd.concat([s1, s2, s3], axis=1)
    return df.idxmax(axis=1)
