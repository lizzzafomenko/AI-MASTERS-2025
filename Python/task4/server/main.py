from fastapi import FastAPI, HTTPException
from concurrent.futures import ProcessPoolExecutor
from pydantic import BaseModel
import numpy as np
import asyncio
from .models import *
from .process_env import *
from multiprocessing import Value, Lock

import time


app = FastAPI(title="ML MODELS INFERENCE SERVER")

cores_for_fit = core_nums - 1
fit_pool = ProcessPoolExecutor(max_workers=cores_for_fit)


# https://blog.jetbrains.com/de/pycharm/2025/08/schnelleres-python-entfernen-des-global-interpreter-lock-in-python/
active_proc_number = Value("i", 0)
lock = Lock()


def fit_job(X, y, config):
        X = np.asarray(X)
        y = np.asarray(y)
        return fit_model(X, y, config)



# как красиво валидировать реквесты: буквально с главной страницы FastAPI: https://fastapi.tiangolo.com/#recap
class FitRequest(BaseModel):
    X: list
    y: list
    config: dict

class PredictRequest(BaseModel):
    X: list
    config: dict

class BasicRequest(BaseModel):
    config: dict


@app.get('/')
def root():
    return {'message': 'Server for models inference is running!', 'models will be saved to': dir_path,
            'maximum fit processes': cores_for_fit, 'maximum models to load': max_models}

@app.get("/status")
def status():
    with lock:
        active_processees = active_proc_number.value 
    
    return {'maximum fit processes': cores_for_fit, 'active processes': active_processees, 'empty processes': cores_for_fit - active_processees}

# https://medium.com/@AlexanderObregon/understanding-pythons-multiprocessing-module-744dba8d4be4
@app.post("/fit")
async def fit_endpoint(request: FitRequest):
    loop = asyncio.get_running_loop()

    with lock:
        active_proc_number.value += 1
    try:
        result = await loop.run_in_executor(
            fit_pool,
            fit_job,
            request.X,
            request.y,
            request.config,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        with lock:
            active_proc_number.value -= 1


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    try:
        result = predict_model(request.X, request.config)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/load")
def load_endpoint(request: BasicRequest):
    try:
        result = load_model(request.config)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/unload")
def unload_endpoint(request: BasicRequest):
    try:
        result = unload_model(request.config)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/remove")
def remove_endpoint(request: BasicRequest):
    try:
        result = remove_model(request.config)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/remove_all")
def remove_all_endpoint():
    try:
        result = remove_all_models()
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))