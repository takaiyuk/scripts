def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'features/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test

feats = ['FamilySize', 'AgeGroup']
X_train, X_test = load_datasets(feats)