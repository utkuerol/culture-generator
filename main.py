from flask import Flask, jsonify, request
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)
query_param_temperature_level = "TemperatureLevel"
query_param_temperature_variance = "TemperatureVariance"
query_param_precipitation_level = "PrecipitationLevel"
query_param_civilization_level = "CivilizationLevel"


@app.route('/culture-generator/api/predict', methods=['GET'])
def get_tasks():
    args = request.args
    temperature_level = args.get(
        query_param_temperature_level, default=-1, type=int)
    temperature_variance = args.get(
        query_param_temperature_variance, default=-1, type=int)
    precipitation_level = args.get(
        query_param_precipitation_level, default=-1, type=int)
    civilization_level = args.get(
        query_param_civilization_level, default=-1, type=int)
    input = [temperature_level, temperature_variance,
             precipitation_level, civilization_level]
    if -1 in input:
        return jsonify({"error": "invalid request"})

    model.train()
    result = model.predict(input)
    return jsonify({'predictions': result})


if __name__ == '__main__':
    app.run(port=8080, debug=True)
