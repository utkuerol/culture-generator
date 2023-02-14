trained_model_file = "model.pkl"
culture_data_path = "data/dplace/EA/data.csv"
eco_data_path = "data/dplace/ecoClimate/data.csv"
columns = ["soc_id", "var_id", "code"]
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
