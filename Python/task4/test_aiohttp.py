import asyncio
import aiohttp
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


# # https://docs.aiohttp.org/en/stable/client_quickstart.html


async def fetch(session, method, endpoint, json_data=None):
    async with session.request(method, f"{url}/{endpoint}", json=json_data) as resp:
        try:
            return resp.status, await resp.json()
        except:
            return resp.status, await resp.text()

async def fit_model(session, config, X_train, y_train):
    model = {'X': X_train.tolist(), 'y': y_train.tolist(), 'config': config}
    start_time = time.time()
    status_code, result = await fetch(session, 'POST', 'fit', model)
    end_time = time.time()
    if status_code == 200:
        print(json.dumps(result, indent=5), '\n')
        print(f'time to fit the model: {end_time - start_time:.3f}', '\n')
    else:
        print(f'error: {result}')
    status_s, result_s = await fetch(session, 'GET', 'status')
    print('SERVER STATUS')
    print(json.dumps(result_s, indent=5), '\n')
    return end_time - start_time


async def get_predictions(session, config, X_test, y_test, prev_config=None):
    status_code, result = await fetch(session, 'POST', 'load', {'config': config})
    if status_code != 200:
        await fetch(session, 'POST', 'unload', {'config': prev_config})
        status_code, result = await fetch(session, 'POST', 'load', {'config': config})
    
    print(json.dumps(result, indent=5), '\n')

    status_code, y_pred = await fetch(session, 'POST', 'predict', {'X': X_test.tolist(), 'config': config})
    print(f'Pearson correlation for {config["name"]}: {pearsonr(y_test, y_pred).statistic:.4f}', '\n')


timeout = aiohttp.ClientTimeout(total=10000)
async def pipeline():
    async with aiohttp.ClientSession(timeout = timeout) as session:
        status_code, result = await fetch(session, 'GET', '')
        print('ROOT SERVER INFO')
        print(json.dumps(result, indent=5), '\n')

        status_code, result = await fetch(session, 'GET', 'status')
        print('SERVER STATUS INFO')
        print(json.dumps(result, indent=5), '\n')

        print('-'*40, 'FIT MODELS', '-'*40, '\n')
        start_time = time.time()
        tasks = [fit_model(session, config, X_train, y_train) for config in configs]
        await asyncio.gather(*tasks)
        end_time = time.time()
        total_time = end_time - start_time

        print('-'*(len('TOTAL TIME for models fitting with request: ') + len(str(total_time)) + 4))
        print(f'| TOTAL TIME for models fitting with request: {total_time} |')
        print('-'*(len('TOTAL TIME for models fitting with request: ') + len(str(total_time)) + 4), '\n\n')


        print('-'*40, 'MAKE PREDICTIONS', '-'*40, '\n')
        prev_config = None
        for config in configs:
            await get_predictions(session, config, X_test, y_test, prev_config)
            prev_config = config
        
        status, result = await fetch(session, 'POST', 'remove', {'config': configs[0]})
        if status == 200:
            print(json.dumps(result, indent=5), '\n')
        else:
            print(f'error: {result}')

        print('removing all the models.....')
        status, result = await fetch(session, 'POST', 'remove_all')
        print(json.dumps(result, indent=5), '\n')

asyncio.run(pipeline())