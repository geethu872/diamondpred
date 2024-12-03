import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder


with open('label1.pkl', 'rb') as f:
    label_encoder1 = pickle.load(f)
with open('label2.pkl', 'rb') as f:
    label_encoder2 = pickle.load(f)
with open('label3.pkl', 'rb') as f:
    label_encoder3 = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


model = pickle.load(open('xgb.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/home1', methods=['POST'])
def home1():
    try:
        carat = float(request.form['a'])
        cut = request.form['b']
        color = request.form['c']
        clarity = request.form['d']
        # depth = float(request.form['e'])
        table = float(request.form['f'])
        length = float(request.form['g'])
        width = float(request.form['h'])
        depth_mm = float(request.form['i'])
        depth1=float((2*depth_mm)/(length+width))
        
        
        cut_encoded = label_encoder2.transform([cut])[0]
        color_encoded = label_encoder3.transform([color])[0]
        clarity_encoded = label_encoder1.transform([clarity])[0]
        
        input_data = np.array([[carat, cut_encoded, color_encoded, clarity_encoded,
                                depth1, table, length, width, depth_mm]])

        
        input_data_scaled = scaler.transform(input_data)

        
        prediction = model.predict(input_data_scaled)[0]

        return render_template('after.html', prediction=f"${prediction}")

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)

