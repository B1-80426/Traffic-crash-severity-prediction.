
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open('/home/sunbeam/Desktop/Project/backend/models/lg.pkl', 'rb') as file:
    model = pickle.load(file)

label_encoders = {
    "weather_1": LabelEncoder(),
    "road_surface": LabelEncoder(),
    "road_cond_1": LabelEncoder(),
    "road_cond_2": LabelEncoder(),
    "lighting": LabelEncoder(),
    "type_of_collision": LabelEncoder(),
    "party1_type": LabelEncoder(),
    "party2_type": LabelEncoder(),
    "party1_dir_of_travel": LabelEncoder(),
    "party2_dir_of_travel": LabelEncoder()
}

options = {
    "party1_type": ['Driver', 'Bicyclist', 'Parked Vehicle', 'Pedestrian', 'Other', 'Not Stated', 'Bicycle'],
    "party2_type": ['Bicyclist', 'Driver', 'Pedestrian', 'Parked Vehicle', 'Other', 'Not Stated'],
    "lighting": ['Daylight', 'Not Stated', 'Dark - Street Lights', 'Dark - Street Lights Not Functioning',
                 'Dusk - Dawn', 'Dark - No Street Lights'],
    "type_of_collision": ['Other', 'Broadside', 'Vehicle/Pedestrian', 'Not Stated', 'Rear End', 'Head-On', 'Hit Object',
                          'Sideswipe', 'Overturned'],
    "road_surface": ['Dry', 'Not Stated', 'Wet', 'Slippery'],
    "weather_1": ['Clear', 'Cloudy', 'Raining', 'Fog', 'Wind',
                  'Snowing'],
    "road_cond_1": ['No Unusual Condition', 'Other',
                    'Construction or Repair Zone', 'Holes, Deep Ruts',
                    'Obstruction on Roadway', 'Holes, Deep Rut',
                    'Reduced Roadway Width', 'Loose Material on Roadway', 'Flooded'],
    "road_cond_2": ['Not Stated', 'No Unusual Condition',
                    'Construction or Repair Zone', 'Flooded', 'Reduced Roadway Width',
                    'Other', 'Obstruction on Roadway'],
    "party1_dir_of_travel": ['East', 'North', 'South', 'West', 'Not Stated'],
    "party2_dir_of_travel": ['East', 'South', 'West', 'North', 'Not Stated']
}

for key, value in options.items():
    label_encoders[key].fit(value)


@app.route("/", methods=["GET"])
def root():
    return render_template('index.html', options=options)


@app.route("/classify", methods=["POST"])
def classify():
    party1_type = request.form.get("party1_type")
    party2_type = request.form.get("party2_type")
    lighting = request.form.get("lighting")
    type_of_collision = request.form.get("type_of_collision")
    road_surface = request.form.get("road_surface")

    weather_1 = request.form.get("weather_1")
    road_cond_1 = request.form.get("road_cond_1")
    road_cond_2 = request.form.get("road_cond_2")
    party1_dir_of_travel = request.form.get("party1_dir_of_travel")
    party2_dir_of_travel = request.form.get("party2_dir_of_travel")

    party1_type_encoded = label_encoders['party1_type'].transform([party1_type])
    party2_type_encoded = label_encoders['party2_type'].transform([party2_type])
    lighting_encoded = label_encoders['lighting'].transform([lighting])
    type_of_collision_encoded = label_encoders['type_of_collision'].transform([type_of_collision])
    road_surface_encoded = label_encoders['road_surface'].transform([road_surface])

    weather_1_encoded = label_encoders['weather_1'].transform([weather_1])
    road_cond_1_encoded = label_encoders['road_cond_1'].transform([road_cond_1])
    road_cond_2_encoded = label_encoders['road_cond_2'].transform([road_cond_2])
    party1_dir_of_travel_encoded = label_encoders['party1_dir_of_travel'].transform([party1_dir_of_travel])
    party2_dir_of_travel_encoded = label_encoders['party2_dir_of_travel'].transform([party2_dir_of_travel])

    prediction = model.predict([
        [
            party1_type_encoded[0], party2_type_encoded[0], type_of_collision_encoded[0], lighting_encoded[0],
            road_surface_encoded[0], weather_1_encoded[0], road_cond_1_encoded[0],
            road_cond_2_encoded[0], party1_dir_of_travel_encoded[0], party2_dir_of_travel_encoded[0]
        ]
    ])

    if prediction[0] == 1:
        return "Prediction: Fatality likely."
    else:
        return "Prediction: Fatality unlikely."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
