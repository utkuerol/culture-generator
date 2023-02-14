from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import preprocess
import shared

print("training the model...")

X, y = preprocess.get_train_data()
rfc = RandomForestClassifier(n_estimators=100)
clf = MultiOutputClassifier(rfc).fit(X, y)
joblib.dump(clf, shared.trained_model_file)
