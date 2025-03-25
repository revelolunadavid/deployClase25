import pickle
from fastapi import APIRouter
import numpy as np
from interfaces import Diabetesdata

router = APIRouter()


with open("RFDiabetesv132.pkl",'rb') as file:
    model = pickle.load(file)

labels = ['Sano','Diabetes']   

@router.get("/")
def index():
    return {"Mensaje": "API 3 Running"}



@router.post('/predict')
def predict_diabetes(data: Diabetesdata):
    data = data.model_dump()
    Pregnancies=data["Pregnancies"]
    Glucose=data["Glucose"]
    BloodPressure=data["BloodPressure"]
    SkinThickness=data["SkinThickness"]
    Insulin=data["Insulin"]	
    BMI=data["BMI"]	
    DiabetesPedigreeFunction=data["DiabetesPedigreeFunction"]	
    Age=data["Age"]

    xin = np.array([Pregnancies,Glucose,BloodPressure,
                    SkinThickness,Insulin,BMI,
                    DiabetesPedigreeFunction,Age]).reshape(1,8)
    prediction = model.predict(xin)
    yout = labels[prediction[0]]
    print('API3, paciente posiblemente {}'.format(yout))
    return {'prediction':yout}


