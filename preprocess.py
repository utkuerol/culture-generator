import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import shared


def _get_data():
    df_culture = pd.read_csv(shared.culture_data_path)
    df_culture = df_culture[df_culture["var_id"].isin(
        shared.culture_variables)]
    df_eco = pd.read_csv(shared.eco_data_path)
    df_eco = df_eco[df_eco["soc_id"].isin(_get_societies_list())]
    df_eco = df_eco[df_eco["var_id"].isin(shared.eco_variables)]

    df_merged = df_culture.append(df_eco)
    df_merged = df_merged[shared.columns]
    df_merged = df_merged.sort_values(by="soc_id")
    df_merged = df_merged.pivot(
        index='soc_id', columns='var_id', values='code')
    df_merged.columns.name = None
    df_merged = df_merged.reset_index()
    df_merged = df_merged.iloc[:, 1:]
    return df_merged


def _clean_data():
    data = _get_data()
    data[shared.eco_variables] = MinMaxScaler(
        feature_range=(0, 9)).fit_transform(data[shared.eco_variables])
    data[shared.eco_variables] = data[shared.eco_variables].astype(int)
    data = data.fillna(-1)
    data[shared.culture_variables] = data[shared.culture_variables].astype(int)
    data = data.rename(columns=dict(
        zip(shared.eco_variables, shared. eco_variables_renamed)))
    return data


def get_train_data():
    data = _clean_data()
    X = data[shared.given_features].to_numpy()
    y = data[shared.target_features].to_numpy()

    return X, y


def _get_societies_list():
    df_societies = pd.read_csv("data/dplace/EA/societies.csv")
    societies_list = df_societies['id'].to_list()
    return societies_list
