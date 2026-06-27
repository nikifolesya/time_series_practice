# Прогнозирование и обнаружение аномалий во временном ряду

Учебный проект по анализу температуры промышленной машины из набора Numenta Anomaly Benchmark (NAB). Проект выполняет полный цикл: загрузка и очистка данных, визуальный анализ, прогнозирование сезонной базовой моделью и SARIMA, обнаружение аномалий робастным методом и Isolation Forest, оценка по официальным окнам NAB.

## Основные результаты

- После объединения 12 повторных временных меток получен регулярный пятиминутный ряд из 22 683 наблюдений без пропусков.
- SARIMA показала RMSE 22,321, сезонная базовая модель - 22,328. Разница мала; по MAPE базовая модель лучше (19,88% против 26,09%).
- Isolation Forest обнаружил сигналы во всех 4 официальных окнах NAB: precision 0,608, recall 0,183, F1 0,281.
- Робастный детектор точнее по отдельным сигналам (precision 0,710), но охватил 2 из 4 окон.

Невысокий точечный recall связан в том числе с форматом разметки: NAB задаёт продолжительные окна событий, тогда как методы отмечают наиболее необычные точки внутри этих окон.

## Структура

```text
.
├── data/
│   ├── raw/                 # исходный CSV и официальные окна NAB
│   └── processed/           # очищенный ряд с признаками и результатами
├── notebooks/
│   └── anomaly_detection_project.ipynb
├── reports/
│   ├── practice_report.pdf
│   ├── practice_diary.pdf
│   └── practice_diary_template.md
├── results/
│   ├── figures/             # итоговые графики
│   ├── forecast_metrics.csv
│   ├── anomaly_metrics.csv
│   └── ...
├── src/
│   ├── pipeline.py          # основной аналитический конвейер
│   ├── build_notebook.py
│   └── build_reports.py
└── requirements.txt
```

## Быстрый запуск

Требуется Python 3.10 или новее.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/pipeline.py
python src/build_notebook.py
python src/build_reports.py
```

Для просмотра ноутбука:

```bash
jupyter lab notebooks/anomaly_detection_project.ipynb
```

Ноутбук уже содержит рассчитанные результаты. Его можно выполнить заново целиком из корня проекта.

## Данные

- Официальный источник: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB).
- Используемый ряд: `data/realKnownCause/machine_temperature_system_failure.csv`.
- Разметка: `labels/combined_windows.json`, четыре окна аномалий для выбранного ряда.

Файлы уже размещены в `data/raw/`, поэтому интернет для повторного анализа не требуется. Подробности приведены в `data/README.md`.

## Итоговые материалы

- [Ноутбук](notebooks/anomaly_detection_project.ipynb)
- [Отчёт](reports/practice_report.pdf)
- [Дневник](reports/practice_diary.pdf)
- Метрики прогноза: `results/forecast_metrics.csv`
- Метрики обнаружения: `results/anomaly_metrics.csv`

## Перед сдачей

1. Заполнить ФИО, группу, руководителя и ссылку на GitFlic на титульном листе отчёта.
2. Заполнить личные данные, сроки и подписи в дневнике либо перенести готовое содержание в утверждённый вузом бланк.
3. Создать личный репозиторий GitFlic, загрузить содержимое проекта и вставить ссылку в отчёт.
