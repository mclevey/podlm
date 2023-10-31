import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_sentiment_distributions(df_long: pd.DataFrame, plot_title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.ecdfplot(data=df_long, x="probability", hue="sentiment")
    ax.set_title(plot_title)
    ax.set_xlabel('Probability of Sentiment Label')
    ax.set_ylabel('Proportion of Comments')
    plt.savefig(filename, dpi=600)
    

def sentiment_small_multiples(ldf_dict: dict, r: str, filename: str):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)

    ax = axes[0, 0]
    sns.ecdfplot(data=ldf_dict['sentences-submission'], x="probability", hue="sentiment", ax=ax, legend=True)
    ax.set_title(f'Sentence-level\nr/{r} Submissions')
    ax.set_xlabel('Probability of Sentiment Label')
    ax.set_ylabel('Proportion')

    ax = axes[0, 1]
    sns.ecdfplot(data=ldf_dict['sentences-comments'], x="probability", hue="sentiment", ax=ax, legend=False)
    ax.set_title(f'Sentence-level\nr/{r} Comments')
    ax.set_xlabel('Probability of Sentiment Label')
    ax.set_ylabel('Proportion')

    ax = axes[1, 0]
    sns.ecdfplot(data=ldf_dict['posts-submission'], x="probability", hue="sentiment", ax=ax, legend=False)
    ax.set_title(f'Post-level\nr/{r} Submissions')
    ax.set_xlabel('Probability of Sentiment Label')
    ax.set_ylabel('Proportion')

    ax = axes[1, 1]
    sns.ecdfplot(data=ldf_dict['posts-comments'], x="probability", hue="sentiment", ax=ax, legend=False)
    ax.set_title(f'Post-level\nr/{r} Comments')
    ax.set_xlabel('Probability of Sentiment Label')
    ax.set_ylabel('Proportion')

    plt.savefig(f'../output/{filename}.png', dpi=600)
    
    
def plot_time_series(df, filename, title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df.values)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()    
    plt.savefig(f'../output/{filename}.png', dpi=600)
    
    
def plot_grouped_count(df, group_col, drop_minus_1=True, plot_title=None, filename=None):
    if drop_minus_1:
        df = df[df[group_col] != -1]
    grouped = df.groupby(group_col).size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(23, 15))
    sns.barplot(x=group_col, y='count', data=grouped, ax=ax, color='slategray')
    ax.set_xlabel(group_col)
    ax.set_ylabel('Count')
    if plot_title:
        fig.suptitle(plot_title, fontsize=16)
    if filename:
        plt.savefig(f'../output/{filename}.png', dpi=600)