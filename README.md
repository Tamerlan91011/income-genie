# Income Genie
Виртуальный ассистент, предсказывающий доходы ваших клиентов

## Команда "Поволжские гоблины"
- [Мокин Дмитрий](https://github.com/Dimires) - ML/DS
- [Елсуков Сергей](https://github.com/DudeDabbler) - ML/DS
- [Глазунов Тимур](https://github.com/Tamerlan91011) - Backend 

## Структура проекта
```bash
income-genie
├── docs/                       # Файлы с постановкой задачи
├── catboost_info/              # Информация о catboost модели
├── saved_models/               # Сохраненный модели после обучения
├── src
│   ├── core
│   │   ├── train.py            # Весь pipeline обучения моделей
│   │   └── inference.py        # Вызов инференца сохраненных моделей
│   ├── datasets                # Использованные в обучении датасеты, предоставленные в рамках хакатона
│   │   └── ...
│   ├── pages                   # Код для Streamlit страниц
│   │   ├── home.py             # 1. Домашняя страница команды
│   │   ├── data_analysis.py    # 2. Анализ данных
│   │   ├── model_training.py   # 3. Описание процесса обучения
│   │   └── training_results.py # 4. Результаты обучения модели
│   │   ├── predictions.py      # 5. Страница использования моделей
│   ├── main.py                 # Точка входа в программу
│   ├── utils.py                # Вспомогательные функции, необходимые при работе со Streamlit
├── ...                         # Прочие файлы
```

## Использованные библиотеки

```bash
# Python 3.11
income-genie v0.1.0
├── catboost v1.2.8
├── joblib v1.5.2
├── lightgbm v4.6.0
├── matplotlib v3.10.7
├── pandas v2.3.3
├── plotly v6.5.0
├── ruff v0.14.6
├── scikit-learn v1.7.2
├── streamlit v1.51.0
└── xgboost v3.1.2
```

## Сборка и запуск приложения
Это веб приложение, построенное с помощью вреймворка Streamlit.
```bash
docker compose up --remove-orphans --build --force-recreate -d
```

## После сборки
Как только контейнеры будут собраны, можно открывать браузер и в адресную строку вводить:
```bash
http://localhost:8080⁠
localhost:8080⁠ # или это, если при попытке перейти по ссылке выше открывается поисковик
```

## Скринкаст
Видео лежит [тут](https://drive.google.com/file/d/1g4_9YRnEmx41_Psns3UrsaO4FV6OIrvW/view?usp=drive_link)
