import pandas as pd
import graph_tool.all as gt
import numpy as np


def save_gt(g, filename):
    g.save(f"../output/{filename}.gt")


def load_gt(filename):
    return gt.load_graph(f"../input/{filename}.gt")


def add_vertex_names_to_edf(vdf, edf, drop_duplicates=False):
    """
    Why are there duplicates here at all? There has to be an issue with the timestamp
    stuff, because it drops down from > 4K edges to ~ 500 edges, which makes me think
    that all the edges are being collapsed into one per author, which is could
    only be the case because they all have the same timestamp (which they should not).
    Will have to go back and look at this later, because for now I need to finish this
    part of the MVP...

    I THINK I GOT IT! I think it's because we split into sentences, and all sentences that
    are part of the same post will have the same timestamp. So an ij interaction will have
    a lot of duplicate edges, because it will have one edge for each sentence in the post!
    I can drop duplicates from this dataframe to get a single unique timestamp per ij
    and then when I query the text dataframe, I can concatenate all the sentences that match
    that specific timestamp, or come before it. - John
    """
    lookup_dict = {}
    for vid, auth in zip(vdf["vid"], vdf["author"]):
        lookup_dict[vid] = auth
    edf["source_author"] = edf["source"].map(lookup_dict)
    edf["target_author"] = edf["target"].map(lookup_dict)
    if drop_duplicates is True:
        edf.drop_duplicates(inplace=True)
    return edf


def get_graph_dfs(g, vtype: int = None, etype: int = None, drop_duplicates=True):
    """
    This function returns two dataframes: one for vertices and one for edges.
    The idea is to use this to compute ij similarities for a given graph.
    """

    vdf = []
    for v in g.vertices():
        vdf.append([int(v), int(g.vp.vtype[v]), str(g.vp.ids[v])])
    vdf = pd.DataFrame(vdf)
    vdf.columns = ["vid", "vtype", "author"]
    if vtype is not None:
        vdf = vdf[vdf["vtype"] == vtype]

    edf = []
    for e in g.edges():
        edf.append(
            [int(e.source()), int(e.target()), int(g.ep.etype[e]), int(g.ep.time[e])]
        )
    edf = pd.DataFrame(edf)
    edf.columns = ["source", "target", "etype", "etime"]
    if etype is not None:
        edf = edf[edf["etype"] == etype]
    edf["datetime"] = pd.to_datetime(edf["etime"], unit="s", utc=True)
    if drop_duplicates is True:
        edf = add_vertex_names_to_edf(vdf, edf, drop_duplicates=True)
    else:
        edf = add_vertex_names_to_edf(vdf, edf, drop_duplicates=False)
    return vdf, edf


def construct_network(
    df: pd.DataFrame, authorcol: str, idcol: str, parentidcol: str
):  # output_path = None, net_type = 'subreddit'
    # node and edge types:
    # 0 = author
    # 1 = subreddit
    # 2 = topic

    df = df[df[authorcol] != "[deleted]"].copy()
    df["epoch"] = (
        (df["datetime"] - pd.Timestamp("1970-01-01", tz="UTC"))
        .astype("timedelta64[s]")
        .astype("int64")
    )

    df["author_type"] = 0
    df["subreddit_type"] = 1
    df["topic_type"] = 2

    post_auth_dict = dict(zip(df[idcol], df[authorcol]))

    df["author_to"] = df[parentidcol].map(post_auth_dict)
    df.dropna(subset=["author_to"], inplace=True)

    df_add = pd.DataFrame()
    df_add["source"] = df[authorcol].tolist() + df[authorcol].tolist()
    df_add["target"] = df["author_to"].tolist() + df["subreddit"].tolist()
    df_add["type"] = df["author_type"].tolist() + df["subreddit_type"].tolist()
    df_add["epoch"] = df["epoch"].tolist() + df["epoch"].tolist()

    g = gt.Graph(
        list(zip(df_add["source"], df_add["target"], df_add["type"], df_add["epoch"])),
        hashed=True,
        hash_type="string",
        directed=True,
        eprops=[("etype", "int"), ("time", "int")],
    )

    vtype = g.new_vertex_property("int")

    g.vp.vtype = vtype

    auth_list = list(set(df[authorcol].tolist() + df["author_to"].tolist()))
    subreddit_list = list(set(df["subreddit"].tolist()))
    #    topic_list = list(set(df['topic'].tolist()))

    vertex_lookup = dict(zip(g.vp.ids, range(len(list(g.vp.ids)))))
    for v in auth_list:
        g.vp.vtype[vertex_lookup[v]] = 0

    for v in subreddit_list:
        g.vp.vtype[vertex_lookup[v]] = 1

    #    for v in topic_list:
    #        g.vp.vtype[vertex_lookup[v]] = 2

    gt.remove_self_loops(g)

    # if output_path != None:
    #     if net_type == 'subreddit':
    #         g.save(output_path + df.iloc[0]['subreddit'] + '.gt')
    #     else:
    #         g.save(output_path + str(net_type) + '.gt')   # for example, a query id number

    return g


def return_graph_type(g, gtype=0):
    # 0 = author
    # 1 = subreddit
    # 2 = topic

    g1 = gt.GraphView(g, efilt=lambda e: g.ep.etype[e] == gtype)
    g1 = gt.GraphView(g1, vfilt=lambda v: g.vp.vtype[v] == gtype)

    return g1


def create_blockmodel(graph, recs=None, rec_types=None, covars=False, refine=False):
    global bs
    bs = []

    clabel = graph.vp["vtype"]

    def collect_partitions(s):
        global bs
        bs.append(s.get_bs())

    callback = collect_partitions

    if covars == True:
        recs = recs
        rec_types = rec_types
    else:
        recs = []
        rec_types = []

    if refine == "basic":
        states = [
            gt.minimize_nested_blockmodel_dl(
                graph,
                state_args=dict(
                    deg_corr=True,
                    recs=recs,
                    rec_types=rec_types,
                    clabel=clabel,
                    pclabel=clabel,
                ),
            )
            for n in range(10)
        ]
        state = states[np.argmin([s.entropy() for s in states])]
    else:
        state = gt.minimize_nested_blockmodel_dl(
            graph,
            state_args=dict(
                deg_corr=True,
                recs=recs,
                rec_types=rec_types,
                clabel=clabel,
                pclabel=clabel,
            ),
        )

    if refine == "marginals":
        gt.mcmc_equilibrate(
            state,
            force_niter=2000,
            mcmc_args=dict(niter=10),
            callback=collect_partitions,
        )

        pmode = gt.PartitionModeState(bs, nested=True, converge=True)
        pv = pmode.get_marginal(graph)
        graph.vertex_properties["pv"] = pv

        bs = pmode.get_max_nested()
        state = state.copy(bs=bs)

    return state


def get_graph_data(state):
    levels = state.get_levels()
    base_level = levels[0].get_blocks()

    name_list = []
    type_list = []
    block_list = []

    for v in state.g.vertices():
        name_list.append(state.g.vp.ids[v])
        type_list.append(state.g.vp.vtype[v])
        block_list.append(base_level[v])

    data_df = pd.DataFrame()
    data_df["name"] = name_list
    data_df["type"] = type_list
    data_df["block id"] = block_list

    return data_df
