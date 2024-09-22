from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load the model with error handling
try:
    model = joblib.load('random_forest_model.FINAL.pkl')
except KeyError as e:
    print(f"KeyError: {e}")
    model = None
except Exception as e:
    print(f"Unexpected error: {e}")
    model = None

# Define the mapping based on your model's training
prediction_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}

app = Flask(__name__)

# Home route - renders the form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route - handles form submissions and makes predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        age = int(request.form['Age'])
        monthly_inhand_salary = float(request.form['Monthly_Inhand_Salary'])
        interest_rate = float(request.form['Interest_Rate'])
        delay_from_due_date = int(request.form['Delay_from_due_date'])
        num_of_delayed_payment = int(request.form['Num_of_Delayed_Payment'])
        credit_utilization_ratio = float(request.form['Credit_Utilization_Ratio'])
        credit_history_age = int(request.form['Credit_History_Age'])
        payment_behaviour = float(request.form['Payment_Behaviour'])
        monthly_balance = float(request.form['Monthly_Balance'])
        credit_age_years = float(request.form['Credit_Age_Years'])
        
        # Prepare input features for the model using DataFrame
        features = pd.DataFrame({
            'Age': [age],
            'Monthly_Inhand_Salary': [monthly_inhand_salary],
            'Interest_Rate': [interest_rate],
            'Delay_from_due_date': [delay_from_due_date],
            'Num_of_Delayed_Payment': [num_of_delayed_payment],
            'Credit_Utilization_Ratio': [credit_utilization_ratio],
            'Credit_History_Age': [credit_history_age],
            'Payment_Behaviour': [payment_behaviour],
            'Monthly_Balance': [monthly_balance],
            'Credit_Age_Years': [credit_age_years]
        })

        # Make prediction using the model
        prediction_index = model.predict(features)[0]
        
        # Map prediction index to result labels
        prediction = prediction_mapping.get(prediction_index, "Unexpected result")

        # Return the appropriate result page
        if prediction == 'Good':
            return render_template('result.good.html')
        elif prediction == 'Standard':
            return render_template('result.standard.html')
        elif prediction == 'Poor':
            return render_template('result.poor.html')
        else:
            return f"Unexpected result: {prediction}"

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

