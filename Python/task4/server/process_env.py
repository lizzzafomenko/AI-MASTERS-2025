# пример, как юзать dotenv для обработки .env файлов
# https://www.geeksforgeeks.org/python/read-environment-variables-with-python-dotenv/

import os
from dotenv import load_dotenv

load_dotenv()

# загружаем переменные из env и ставим параметры по умолчанию
dir_path = os.getenv("dir_path", "./saved_models")
core_nums = int(os.getenv("core_nums", 5))
max_models = int(os.getenv("max_models", 3))

# создаем директорию для загруженных моделей 
os.makedirs(dir_path, exist_ok=True)