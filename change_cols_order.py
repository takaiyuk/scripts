# change the order of the columns by making specified columns to be the head
def change_cols_order(df, head_cols, new_cols=True):
    """
    - INPUT
      - df: data frame
      - head_cols: cols to be the head
    - OUTPUT
      - df: data frame
      - new_cols: columns whose order have been changed
    """
    if type(head_cols) != list: head_cols = [head_cols]
    cols = df.columns.tolist()
    for col in head_cols:
        cols.remove(col)
    new_cols = head_cols + cols
    df = df.loc[:,new_cols]
    if new_cols == True:
        return df, new_cols
    else:
        return df
