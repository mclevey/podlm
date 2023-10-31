import yaml
import pandas as pd

def load_params(params_path: str ='../input/parameters.yaml'):
    with open(params_path, 'rb') as file:
        params = yaml.safe_load(file)
    return params


def load_parquet(filename: str):
    return pd.read_parquet(f'../input/{filename}.parquet.gzip')


def save_parquet(df: pd.DataFrame, filename: str):
    df.to_parquet(f'../output/{filename}.parquet.gzip', compression='gzip')
    print(f'Wrote {filename}.parquet.gzip to ../output/')
    

def merge_string_columns(df: pd.DataFrame, col1: pd.Series, col2: pd.Series, new_col_name: str, drop=True):
    df[col1].replace('[deleted]', '', inplace=True)
    df[col2].replace('[deleted]', '', inplace=True)
    df[new_col_name] = df[col1].fillna('') + df[col2].fillna('')
    if drop is True:
        df = df.drop(columns=[col1, col2])
    return df


def reshape_sentiment_df(df: pd.DataFrame, id_vars: str = 'id', value_vars = ['P(negative)', 'P(neutral)', 'P(positive)']):
    df_long = pd.melt(df, id_vars, value_vars)
    df_long.columns = ['id', 'sentiment', 'probability']
    return df_long


def merge_feature_dfs(df_text_tasks, df_as_collected, on_cols):
    merged = pd.merge(df_text_tasks, df_as_collected, on=on_cols)
    return merged


def group_and_aggregate_author_level(df, group_cols, sum_cols, str_cols, join_str_with=' '):
    grouped = df.groupby(group_cols, as_index=False)
    summed = grouped[sum_cols].sum()
    for col in str_cols:
        concat_concat = grouped[col].agg(lambda x: join_str_with.join(x))   
        summed[col + '_concat'] = concat_concat[col]
    return summed
    
    
def get_largest_series(s1, s2, s3):
    df = pd.concat([s1, s2, s3], axis=1)
    return df.idxmax(axis=1)