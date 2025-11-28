# ЭТО ТОЧКА ВХОДА ДЛЯ ПРИЛОЖЕНИЯ НА FASTAPI

import sys
from pathlib import Path

from fastapi import FastAPI
import uvicorn

sys.path.append(str(Path(__file__).parent.parent))
# ПОСЛЕ ЭТОЙ СТРОКИ ИМПОРТИРУЕТСЯ ВСЕ, 
# ЧТО БЫЛО НАПИСАНО НАМИ В ДАННОМ ПРОЕКТЕ

app = FastAPI()

if __name__ == "__main__":
    # Тут пока ничего нет. Приложение пустое
    uvicorn.run("main:app", reload=True)
