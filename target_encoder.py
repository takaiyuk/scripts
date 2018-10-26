## https://www.slideshare.net/HJvanVeen/feature-engineering-72376750

def target_encode_naive(df, COLUMN, TARGET, train_rate=0.7, shuffle_data=True, SEED=42):
    """
    - INPUT
        - df: pd.DataFrame()
        - COLUMN: column to be encoded
        - TARGET: target variable
        - train_rate: the rate of the length of train dataset
        - shuffle_data: whether or not to shuffle data
        - SEED: random seed when shuffling data
    - OUTPUT:
        - X_train: encoded features of train dataset
        - y_train: target variable of train dataset
        - X_test: encoded features of test dataset
        - y_test: target variable of test dataset
    """
    tr_length = round(df.shape[0] * train_rate)
    if shuffle_data:
        np.random.seed(SEED)
        tr_idx = np.random.choice(range(df.shape[0]), tr_length)

    df_train = df.iloc[:tr_length, :]
    df_test = df.iloc[tr_length:, :]

    df_agg = df_train.groupby(COLUMN)[TARGET].mean().reset_index()
    df_agg.rename(columns={TARGET: COLUMN+"_mean"}, inplace=True)

    df_train = df_train.merge(df_agg, on=COLUMN, how="left")
    X_train = df_train.drop(TARGET,axis=1)
    y_train = df_train[TARGET]
    del df_train

    df_test = df_test.merge(df_agg, on=COLUMN, how="left")
    X_test = df_test.drop(TARGET,axis=1)
    y_test = df_test[TARGET]
    del df_test, df_agg

    return X_train, y_train, X_test, y_test



## https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode_smoothing(tr_series=None, te_series=None, target=None,
                            min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    tr_series : training categorical feature as a pd.Series
    te_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(tr_series) == len(target)
    assert tr_series.name == te_series.name

    df_tmp = pd.concat([tr_series, target], axis=1)

    # Compute target mean
    df_agg = df_tmp.groupby(by=tr_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(df_agg["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data
    prior = target.mean()

    # The bigger the count the less full_avg is taken into account
    df_agg[target.name] = prior * (1 - smoothing) + df_agg["mean"] * smoothing
    df_agg.drop(["mean", "count"], axis=1, inplace=True)

    # Apply average function to trn and tst series
    ft_tr_series = pd.merge(tr_series.to_frame(tr_series.name),
                            df_agg.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
                            on=tr_series.name,
                            how='left')['average'].rename(tr_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_tr_series.index = tr_series.index
    ft_te_series = pd.merge(te_series.to_frame(te_series.name),
                            df_agg.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
                            on=te_series.name,
                            how='left')['average'].rename(tr_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_te_series.index = te_series.index

    return add_noise(ft_tr_series, noise_level), add_noise(ft_te_series, noise_level)
