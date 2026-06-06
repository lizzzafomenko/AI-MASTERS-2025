# test.py
import requests
import time
import json

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr


X, y = make_regression(n_samples = 1500000,
                        n_features = 50, 
                        random_state = 57, 
                        noise = 0.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 57, test_size = 0.2)


url = "http://127.0.0.1:8000"

resp = requests.get(f"{url}/")
print(json.dumps(resp.json(), indent = 5), '\n') 

resp = requests.get(f"{url}/status")
print(json.dumps(resp.json(), indent = 5), '\n') 

# here we assign configa information for our future models

linreg_config = {'name': 'LinearRegression',
                'model': 'linear_reg',
                'fit_intercept': True}

sgdreg_config = {'name': 'SGDRegressor',
                'model': 'sgd_reg',
                'tol': 1e-6,
                'random_state': 57,
                'penalty': 'l2',
                'alpha': 0.05}

dectree_config = {'name': 'DecisionTreeRegressor',
                'model': 'tree_reg',
                'max_depth': 10,
                'min_samples_leaf': 2,
                'random_state': 57}

rforest_config = {'name': 'RandomForestRegressor',
                'model': 'forest_reg',
                'max_depth': 5,
                'n_estimators':20,
                'random_state': 57}

gboost_config = {'name': 'GradientBoostingRegressor',
                'model': 'gb_reg',
                'learning_rate': 0.05,
                'n_estimators': 20,
                'max_depth': 5,
                'random_state': 57}


configs = [linreg_config, sgdreg_config, dectree_config, rforest_config, gboost_config]


# fit models using REQUEST

print('-'*40, 'FIT MODELS', '-'*40, '\n')
total_time = 0

for config in configs:
    model = {'X': X_train.tolist(), 'y': y_train.tolist(), 'config': config}
    start_time = time.time()
    resp = requests.post(f"{url}/fit", json=model)
    end_time = time.time()
    if resp.status_code == 200:
        print(json.dumps(resp.json(), indent = 5), '\n')
        print(f'time to fit the model: {end_time - start_time:.3f}', '\n')
        total_time += end_time - start_time
    else:
        print(f'error: {resp.text}')

    resp = requests.get(f"{url}/status")
    print('SERVER STATUS')
    print(json.dumps(resp.json(), indent = 5), '\n')
    
print('-'*(len('TOTAL TIME for models fitting with request: ') + len(str(total_time)) + 4))
print(f'| TOTAL TIME for models fitting with request: {total_time} |')
print('-'*(len('TOTAL TIME for models fitting with request: ') + len(str(total_time)) + 4), '\n\n')


# predictions
print('-'*40, 'MAKE PREDICTIONS', '-'*40, '\n')

for i in range(len(configs)):
    config = configs[i]
    resp = requests.post(f'{url}/load', json={'config': config})
    if resp.status_code != 200:
        print(f'error: {resp.text}')                  # значит уже все занято, надо что-то удалить
        resp = requests.post(f'{url}/unload', json={'config': configs[i-1]})
        print(json.dumps(resp.json(), indent = 5), '\n')  
        resp = requests.post(f'{url}/load', json={'config': config})

    print(json.dumps(resp.json(), indent = 5), '\n')  
    predict_data = {'X': X_test.tolist(), 'config': config}
    resp = requests.post(f'{url}/predict', json = predict_data)
    y_pred = resp.json()
    print(f'Pearson correlation for {config["name"]}: {pearsonr(y_test, y_pred).statistic:.4f}', '\n')

print('\n')
resp = requests.post(f'{url}/remove', json={'config': configs[0]})
if resp.status_code == 200:
    print(json.dumps(resp.json(), indent = 5), '\n') 
else:
    print(f'error: {resp.text}')

print('removing all the models.....')

resp = requests.post(f"{url}/remove_all")
print(json.dumps(resp.json(), indent = 5), '\n') 

