# import pickle
# import numpy as np
# from flask import Flask, render_template, request

# # Load pre-trained models and encoders
# with open('season.pkl', 'rb') as f:
#     label_seas = pickle.load(f)
# with open('wea_type.pkl', 'rb') as f:
#     label_wea = pickle.load(f)
# with open('cloud.pkl', 'rb') as f:
#     label_cloud = pickle.load(f)
# with open('scalern.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Load your XGBoost model
# model = pickle.load(open('decision.pkl', 'rb'))

# app = Flask(__name__)

# @app.route('/')
# def homepage():
#     return render_template('index.html')

# @app.route('/home1', methods=['POST'])
# def home1():
#     try:
#         # Get data from form
#         temperature = float(request.form['a'])
#         humidity = float(request.form['b'])
#         windspeed = float(request.form['c'])
#         precipitation = float(request.form['d'])
#         cloudcover = request.form['e']
#         atmospre = float(request.form['f'])
#         uv = float(request.form['g'])
#         season = request.form['h']
#         visibility = float(request.form['i'])
        
#         # Perform label encoding
#         seas_encoded = label_seas.transform([season])[0]
#         cloud_encoded = label_cloud.transform([cloudcover])[0]

#         # Create input array for prediction
#         input_data = np.array([[temperature, humidity, windspeed, precipitation, cloud_encoded, atmospre, uv, seas_encoded, visibility]])

#         # Standardize numerical features using the scaler
#         input_data_scaled = scaler.transform(input_data)

#         # Make prediction using the model
#         prediction = model.predict(input_data_scaled)[0]

#         return render_template('result.html', prediction=prediction)

#     except Exception as e:
#         print(f"Prediction error: {str(e)}")
#         return render_template('error.html', error_message=str(e))  # Make sure to create an error.html template

# if __name__ == '__main__':
#     app.run(debug=True)
import pickle
import numpy as np
from flask import Flask, render_template, request

# Load pre-trained models and encoders
with open('season.pkl', 'rb') as f:
    label_seas = pickle.load(f)
with open('wea_type.pkl', 'rb') as f:
    label_wea = pickle.load(f)
with open('cloud.pkl', 'rb') as f:
    label_cloud = pickle.load(f)
with open('scalern.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load your XGBoost model
model = pickle.load(open('decision.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/home1', methods=['POST'])
def home1():
    try:
        # Get data from form
        temperature = float(request.form['a'])
        humidity = float(request.form['b'])
        windspeed = float(request.form['c'])
        precipitation = float(request.form['d'])
        cloudcover = request.form['e']
        atmospre = float(request.form['f'])
        uv = float(request.form['g'])
        season = request.form['h']
        visibility = float(request.form['i'])
        
        # Perform label encoding
        seas_encoded = label_seas.transform([season])[0]
        cloud_encoded = label_cloud.transform([cloudcover])[0]

        # Create input array for prediction
        input_data = np.array([[temperature, humidity, windspeed, precipitation, cloud_encoded, atmospre, uv, seas_encoded, visibility]])

        # Standardize numerical features using the scaler
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the model
        prediction_encoded = model.predict(input_data_scaled)[0]

        # Decode the prediction
        prediction_decoded = label_wea.inverse_transform([prediction_encoded])[0]

        return render_template('result.html', prediction=prediction_decoded)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('error.html', error_message=str(e))  # Make sure to create an error.html template

if __name__ == '__main__':
    app.run(debug=True)

