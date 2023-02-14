import random
import joblib
import pandas as pd
from os.path import exists
import shared

culture_variables_path = "data/dplace/EA/variables.csv"
culture_variables_codes_path = "data/dplace/EA/codes.csv"
readable_result_variable_category = "variable_category"
readable_result_variable_title = "variable_title"
readable_result_variable_definition = "variable_definition"
readable_result_code_name = "code_name"
readable_result_code_description = "code_description"


def predict(input):
    """
    input must be a 1x4 array (TemperatureLevel, TemperatureVariance, PrecipitationLevel, CivilizationLevel)
    """
    input = [input]
    model = _get_saved_model()
    result = model.predict_proba(input)
    result_curated = []
    for i, p in enumerate(result):
        p[0][0] = 0
        tmp = list(p[0])
        tmp.sort(reverse=True)
        noise_level = random.randint(0, 2)
        if shared.target_features[i] in shared.ecology_related_variables:
            noise_level = 0
        selection = tmp[noise_level]
        code = model.classes_[i][list(p[0]).index(selection)]
        while code == -1:
            noise_level = random.randint(0, 2)
            selection = tmp[noise_level]
            code = model.classes_[i][list(p[0]).index(selection)]
        result_curated.append(code)
    result_pprint = []
    for i in range(len(shared.target_features) - 1):
        variable = shared.target_features[i]
        code = result_curated[i]
        result_pprint.append(_pprint(variable, code))
    return result_pprint


def _pprint(variable, code):
    df_vars = pd.read_csv(culture_variables_path)
    df_codes = pd.read_csv(culture_variables_codes_path)
    pprint = dict()
    if variable not in shared.culture_variables:
        return
    found_var = df_vars[df_vars["id"] == variable]
    pprint[readable_result_variable_category] = found_var["category"].values[0]
    pprint[readable_result_variable_title] = found_var["title"].values[0]
    pprint[readable_result_variable_definition] = found_var["definition"].values[0]
    df_codes = df_codes[df_codes["var_id"] == variable]
    found_code = df_codes[df_codes["code"] == code]
    pprint[readable_result_code_name] = found_code["name"].values[0]
    pprint[readable_result_code_description] = found_code["description"].values[0]
    return pprint


def _get_saved_model():
    if exists(shared.trained_model_file):
        clf = joblib.load(shared.trained_model_file)
        return clf
