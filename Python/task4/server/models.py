# для сохранения моделей юзаем pickle, как советует WANDB: https://wandb.ai/a-sh0ts/publications/reports/How-to-Save-a-Classifier-to-Disk-in-Scikit-learn--Vmlldzo0NDc1ODI0 

import os
import pickle  
from .process_env import *
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np


# Словарь для загруженных моделей

loaded_models = {}

MODELS_REGISTRY = {
    # classificators
    'logistic_clf': LogisticRegression,
    'sgd_clf': SGDClassifier,
    'tree_clf': DecisionTreeClassifier,
    'forest_clf': RandomForestClassifier,
    'gb_clf': GradientBoostingClassifier,

    # regressions
    'linear_reg': LinearRegression,
    'sgd_reg': SGDRegressor,
    'tree_reg': DecisionTreeRegressor,
    'forest_reg': RandomForestRegressor,
    'gb_reg': GradientBoostingRegressor,
}


def fit_model(X, y, config):
    cfg = config.copy()

    name = cfg.pop('name')
    model_path = os.path.join(dir_path, f'{name}.pkl')

    X = np.array(X)
    y = np.array(y)
    
    # если имена моделей дублируются, кидаем error и сообщаем о повторке
    if os.path.exists(model_path):
        raise ValueError(f'Unable to create model with name {name}: file already exists')

    # модель = класс из sklearn с fit, fit_tranform, predict методами
    model = cfg.pop('model')
    if model not in MODELS_REGISTRY.keys():
        raise ValueError(f'Unsupported model type')
    model = MODELS_REGISTRY[model](**cfg)

    model.fit(X, y)

    # сохраняем модель
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {'status': 'success', 'message': f'Model {model} with name {name} is fit and saved to {model_path}'}


def predict_model(X, config):
    cfg = config.copy()
    name = cfg.pop('name')

    X = np.array(X)

    if name not in loaded_models:
        raise ValueError(f'Model with name {name} is not loaded')
    
    model = loaded_models[name]
    y_pred = model.predict(X)

    return y_pred.tolist()


def load_model(config):

    if len(loaded_models.keys()) >= max_models:
        raise ValueError(f'Number of loaded models is already the maximum possible')

    name = config['name']

    # если уже загружено, то ниче не делаем, но сообщаем об этом
    if name in loaded_models:
        return {'status': 'warning', 'message': f'Model {name} is already loaded'}

    model_path = os.path.join(dir_path, f'{name}.pkl')

    # запросы с несуществующими именами моделей
    if not os.path.exists(model_path):
        raise ValueError(f'Loading the model {name} from {model_path} failed: no such file')

    # загружаем модель и добавляем ее в loaded_models
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    loaded_models[name] = model
    return {'status': 'success', 'message': f'Model {name} is loaded'}


def unload_model(config):
    name = config['name']

    if name not in loaded_models.keys():
        raise ValueError(f'Model with name {name} is not loaded')

    del loaded_models[name]
    return {'status': 'success', 'message': f'Model {name} is unloaded'}


def remove_model(config):
    name = config['name']
    model_path = os.path.join(dir_path, f'{name}.pkl')

    if not os.path.exists(model_path):
        raise ValueError(f'No such file {model_path}')

    os.remove(model_path)

    if name in loaded_models:
        del loaded_models[name]

    return {'status': 'success', 'message': f'Model {name} removed'}


def remove_all_models():
    for file in os.listdir(dir_path):
        if file.endswith(".pkl"):
            os.remove(os.path.join(dir_path, file))
    loaded_models.clear()
    return {'status': 'success', 'message': f'All models removed'}