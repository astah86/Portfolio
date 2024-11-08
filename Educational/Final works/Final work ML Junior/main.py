import json
import dill


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

file_name = 'final.pkl'
with open(file_name, 'rb') as file:
    object_to_load = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    id: str
    probability_of_success: str
    prediction: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return object_to_load['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame.from_dict([form.dict()])

    model_predict = object_to_load['model'].predict_proba(df)
    probs_val = model_predict[:, 1]
    cl = (probs_val > 0.061).astype(int)

    return {
        'id': form.session_id,
        'probability_of_success': str(np.around(probs_val[0] * 100, 2)) + '%',
        'prediction': cl

    }

