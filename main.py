# Import Libraries
from utils import process_new
import joblib
import os
import numpy as np
from fastapi import FastAPI


# Load the Model
MODEL_PATH = os.path.join(os.getcwd(),'Model_RandomForest.pkl')
model =joblib.load(MODEL_PATH)

# Initialize an app
app = FastAPI()

@app.get('/root')

async def root(fixed_acidity:float,volatile_acidity:float,citric_acid:float,residual_sugar:float,chlorides:float,
               free_sulfur_dioxide:float,total_sulfur_dioxide:float,density:float,pH:float,sulphates:float,alcohol:float):
    
    # Concatenate
    new_data= np.array([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,
                        total_sulfur_dioxide,density,pH,sulphates,alcohol])
    
    # Call the function from utils.py
    X_processed= process_new(X_new=new_data)

    # Model Prediction
    y_pred = model.predict(X_processed)[0]
    y_pred = bool(y_pred)


    return {f'AirLine Classifacation` is {y_pred}'}
