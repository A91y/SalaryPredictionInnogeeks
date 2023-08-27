from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model_xg.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        years_of_experience = float(request.form['years_of_experience'])

        # Make a prediction using the model
        input_data = [[age, gender, years_of_experience]]
        prediction = model.predict(input_data)[0]
        prediction_value = int(round(prediction, 0))*40 # Adjustment to Indian Rupees
        prediction_value = "{:,}".format(prediction_value)
        # Return JSON response
        response = {
            'prediction': f'Predicted Salary: Rs. {prediction_value}',
            'input_data': f'Input Data: Age={age}, Gender={gender}, Years of Experience={years_of_experience}',
        }

        return jsonify(response)
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter valid numerical values.'})

if __name__ == '__main__':
    app.run(debug=True)
