from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    units = float(request.form['units'])
    ac_hours = float(request.form['ac_hours'])
    temperature = float(request.form['temperature'])

    features = np.array([[units, ac_hours, temperature]])

    prediction = model.predict(features)

    result = round(prediction[0], 2)

    if result < 1500:
        suggestion = "Low electricity usage ✅"

    elif result < 3000:
        suggestion = "Moderate usage ⚠️ Try reducing AC usage"

    else:
        suggestion = "High electricity usage ❌ Save energy to reduce bills"

    return render_template(
        'index.html',
        prediction_text=f'Predicted Electricity Bill: ₹ {result}',
        suggestion_text=suggestion
    )

if __name__ == '__main__':
    app.run(debug=True)