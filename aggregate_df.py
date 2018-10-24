## https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/code?scriptVersionId=6025993

# Perform aggregations
def aggregate_df(df, group, cols, aggs):
    """
    INPUT
        df: data frame
        group: list of column to be grouped
        cols: list of columns to be aggregated
        aggs: list of methods to aggregate
    OUTPUT
        df_agg: data frame aggregated
        new_columns: columns of df_agg
    """
    dict_agg = {}
    for col in cols:
        dict_agg.update({col: aggs})
    df_agg = df.groupby(group).agg(dict_agg)
    df_agg.columns = [c1+"_"+c2.upper() for c1, c2 in df_agg.columns]
    new_columns = df_agg.columns.tolist()
    del dict_agg
    return df_agg, new_columns

# Merge df_agg with original data frame
def merge_agg(df, group, cols, aggs, drop_groupcol=True):
    """
    INPUT
        df: data frame
        group: list of column to be groupby
        cols: list of columns to be aggregated
        aggs: list of methods to aggregate
        drop_groupcol: if True, drop columns to be groupby
    OUTPUT
        aggregate data frame and merge it with original data frame
    """
    df_agg, _ = aggregate_df(df, group, cols, aggs)
    df = df.join(df_agg, how='left', on=group)
    if drop_groupcol:
        df.drop([group], axis=1, inplace= True)
    del df_agg
    return df
