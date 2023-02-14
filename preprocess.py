import pandas as pd
from sklearn.preprocessing import MinMaxScaler

culture_data_path = "data/dplace/EA/data.csv"
eco_data_path = "data/dplace/ecoClimate/data.csv"
columns = ["soc_id", "var_id", "code"]
df_societies = pd.read_csv("data/dplace/EA/societies.csv")
societies_list = df_societies['id'].to_list()
eco_variables = [
    "AnnualMeanTemperature",
    "AnnualTemperatureVariance",
    "MonthlyMeanPrecipitation"
]
eco_variables_renamed = [
    "TemperatureLevel",
    "TemperatureVariance",
    "PrecipitationLevel"
]
culture_variables = [
    "EA001",
    "EA002",
    "EA003",
    "EA004",
    "EA005",
    "EA006",
    "EA008",
    "EA009",
    "EA016",
    "EA023",
    "EA028",
    "EA029",
    "EA030",
    "EA031",
    "EA033",
    "EA034",
    "EA035",
    "EA038",
    "EA040",
    "EA042",
    "EA043",
    "EA066",
    "EA068",
    "EA070",
    "EA072",
    "EA078",
    "EA079",
    "EA080",
    "EA081",
    "EA082",
    "EA083",
    "EA113",
]
ecology_related_variables = [
    "EA001",
    "EA002",
    "EA003",
    "EA004",
    "EA005",
    "EA006",
    "EA028",
    "EA029",
    "EA042"
]
all_features = eco_variables_renamed + culture_variables
given_features = eco_variables_renamed + ["EA033"]
target_features = [x for x in all_features if x not in given_features]


def _get_data():
    df_culture = pd.read_csv(culture_data_path)
    df_culture = df_culture[df_culture["var_id"].isin(culture_variables)]
    df_eco = pd.read_csv(eco_data_path)
    df_eco = df_eco[df_eco["soc_id"].isin(societies_list)]
    df_eco = df_eco[df_eco["var_id"].isin(eco_variables)]

    df_merged = df_culture.append(df_eco)
    df_merged = df_merged[columns]
    df_merged = df_merged.sort_values(by="soc_id")
    df_merged = df_merged.pivot(
        index='soc_id', columns='var_id', values='code')
    df_merged.columns.name = None
    df_merged = df_merged.reset_index()
    df_merged = df_merged.iloc[:, 1:]
    return df_merged


def _clean_data():
    data = _get_data()
    data[eco_variables] = MinMaxScaler(
        feature_range=(0, 9)).fit_transform(data[eco_variables])
    data[eco_variables] = data[eco_variables].astype(int)
    data = data.fillna(-1)
    data[culture_variables] = data[culture_variables].astype(int)
    data = data.rename(columns=dict(
        zip(eco_variables, eco_variables_renamed)))
    return data


def get_train_data():
    data = _clean_data()
    X = data[given_features].to_numpy()
    y = data[target_features].to_numpy()

    return X, y
