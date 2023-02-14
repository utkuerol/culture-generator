from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import random
import preprocess
import joblib
import pandas as pd
from os.path import exists

culture_variables_path = "data/dplace/EA/variables.csv"
culture_variables_codes_path = "data/dplace/EA/codes.csv"
readable_result_variable_category = "variable_category"
readable_result_variable_title = "variable_title"
readable_result_variable_definition = "variable_definition"
readable_result_code_name = "code_name"
readable_result_code_description = "code_description"

trained_model_file = "model.pkl"

df_vars = pd.read_csv(culture_variables_path)
df_codes = pd.read_csv(culture_variables_codes_path)
model = None


def train():
    if model is None:
        print("using the existing model...")
        model = _get_saved_model()
        return
    print("training the model...")
    X, y = preprocess.get_train_data()
    rfc = RandomForestClassifier(n_estimators=100)
    clf = MultiOutputClassifier(rfc).fit(X, y)
    joblib.dump(clf, trained_model_file)
    model = clf


def predict(input):
    """ 
    input must be a 1x4 array (TemperatureLevel, TemperatureVariance, PrecipitationLevel, CivilizationLevel) 
    """
    input = [input]
    result = model.predict_proba(input)
    result_curated = []
    for i, p in enumerate(result):
        p[0][0] = 0
        tmp = list(p[0])
        tmp.sort(reverse=True)
        noise_level = random.randint(0, 2)
        selection = tmp[noise_level]
        code = model.classes_[i][list(p[0]).index(selection)]
        while code == -1:
            noise_level = random.randint(0, 2)
            selection = tmp[noise_level]
            code = model.classes_[i][list(p[0]).index(selection)]
        result_curated.append(code)
    result_pprint = []
    for i in range(len(preprocess.target_features) - 1):
        variable = preprocess.target_features[i]
        code = result_curated[i]
        result_pprint.append(_pprint(variable, code))
    return result_pprint


def _pprint(variable, code):
    pprint = dict()
    if variable not in preprocess.culture_variables:
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
    if exists(trained_model_file):
        clf = joblib.load(trained_model_file)
        return clf
