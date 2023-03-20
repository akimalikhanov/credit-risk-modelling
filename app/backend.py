from fastapi import FastAPI, Request
import uvicorn
import config
from utils import *


path_dict=config.PATH_DICT

app = FastAPI()


@app.get("/")
def root():
    return {'Description': 'API for calculating probability of default'}


@app.post("/predict")
async def post_prediction(data: Request):
    cient_data=await data.json()
    prediction=predict_api(cient_data[0],
                            path_dict['dtypes'],
                            path_dict['model_file'],
                            path_dict['lgb_ohe'])
    
    return {
        "prediction" : prediction
    }


if __name__=='__main__':
    uvicorn.run('backend:app', host='0.0.0.0', port=8000, reload=True)

# python -m uvicorn main:app --reload  