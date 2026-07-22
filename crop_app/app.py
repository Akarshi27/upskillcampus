"""
app.py - Crop Yield Prediction Web App
=====================================
This file does 3 things:
1. Loads the dataset and trains the model
2. Creates the web server using Flask
3. Handles predictions from the HTML form
"""

# ── Step 1: Import libraries ──────────────────────────────────
import os
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# ── Step 2: Create Flask app ──────────────────────────────────
app = Flask(__name__)

# ── Step 3: Load dataset ──────────────────────────────────────


base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'datafile (1).csv')

df = pd.read_csv(file_path, encoding='latin1')

# Clean column names
df.columns = df.columns.str.strip()

# Rename long columns to short names
df.rename(columns={
    'Yield (Quintal/ Hectare)'              : 'Yield',
    'Cost of Cultivation (`/Hectare) A2+FL' : 'Cost_A2FL',
    'Cost of Cultivation (`/Hectare) C2'    : 'Cost_C2',
    'Cost of Production (`/Quintal) C2'     : 'CostProd_C2'
}, inplace=True)

# ── Step 4: Encode text columns ───────────────────────────────
le_crop  = LabelEncoder()
le_state = LabelEncoder()
df['Crop_enc']  = le_crop.fit_transform(df['Crop'])
df['State_enc'] = le_state.fit_transform(df['State'])

# ── Step 5: Get dropdown lists ────────────────────────────────
# Define BEFORE train/test split
crop_list  = sorted(df['Crop'].unique().tolist())
state_list = sorted(df['State'].unique().tolist())

# ── Step 6: Prepare features and target ───────────────────────
X = df[['Crop_enc', 'State_enc', 'Cost_A2FL', 'Cost_C2', 'CostProd_C2']]
y = df['Yield']

# ── Step 7: Train/Test Split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Step 8: Train model ───────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)      # Train on 80% only

# ── Step 9: Evaluate model ────────────────────────────────────
y_pred = model.predict(X_test)
r2     = round(r2_score(y_test, y_pred), 4)
mae    = round(mean_absolute_error(y_test, y_pred), 4)

print("✅ Model trained successfully!")
print(f"   Train size : {len(X_train)} rows (80%)")
print(f"   Test size  : {len(X_test)}  rows (20%)")
print(f"   R2 Score   : {r2}")
print(f"   MAE Score  : {mae}")
print(f"   Crops      : {len(crop_list)}")
print(f"   States     : {len(state_list)}")


# ── Step 10: Home page route ──────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html',
                           crops=crop_list,
                           states=state_list,
                           prediction=None,
                           error=None)


# ── Step 11: Predict route ────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from HTML form
        crop         = request.form['crop']
        state        = request.form['state']
        cost_a2fl    = float(request.form['cost_a2fl'])
        cost_c2      = float(request.form['cost_c2'])
        cost_prod_c2 = float(request.form['cost_prod_c2'])

        # Encode crop and state
        crop_enc  = le_crop.transform([crop])[0]
        state_enc = le_state.transform([state])[0]

        # Create input for prediction
        input_data = pd.DataFrame(
            [[crop_enc, state_enc, cost_a2fl, cost_c2, cost_prod_c2]],
            columns=['Crop_enc', 'State_enc',
                     'Cost_A2FL', 'Cost_C2', 'CostProd_C2']
        )

        # Predict
        result     = model.predict(input_data)[0]
        prediction = round(result, 2)

        return render_template('index.html',
                               crops=crop_list,
                               states=state_list,
                               prediction=prediction,
                               selected_crop=crop,
                               selected_state=state,
                               error=None)

    except Exception as e:
        return render_template('index.html',
                               crops=crop_list,
                               states=state_list,
                               prediction=None,
                               error=str(e))


# ── Step 12: Run app ──────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)