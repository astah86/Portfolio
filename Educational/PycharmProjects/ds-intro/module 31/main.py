import json
import dill


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

file_name = 'model/cars_pipe.pkl'
with open(file_name, 'rb') as file:
   object_to_load = dill.load(file)


class Form(BaseModel):
    id: str
    url: str
    region: str
    region_url: str
    price: int
    year: int
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


class Prediction(BaseModel):
    id: int
    pred: str
    price: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return object_to_load['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    import pandas as pd
    df = pd.DataFrame.from_dict([form.dict()])
    y = object_to_load['model'].predict(df)

    return {
        'id': form.id,
        'pred': y[0],
        'price': form.price
    }