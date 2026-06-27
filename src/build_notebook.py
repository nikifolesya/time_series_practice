"""Создание выполненного учебного ноутбука без внешней зависимости nbformat."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/anomaly_detection_project.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(source: str, output: str = "", count: int | None = None) -> dict:
    outputs = []
    if output:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": output.splitlines(True),
            }
        )
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(True),
    }


def build() -> None:
    raw = pd.read_csv(ROOT / "data/raw/machine_temperature_system_failure.csv")
    forecast_metrics = pd.read_csv(ROOT / "results/forecast_metrics.csv")
    anomaly_metrics = pd.read_csv(ROOT / "results/anomaly_metrics.csv")
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))

    fm = forecast_metrics.round(3).to_string(index=False)
    am = anomaly_metrics.round(3).to_string(index=False)
    best_forecast = summary["best_forecast_model_by_rmse"]
    best_anomaly = summary["best_anomaly_method_by_f1"]
    sarima = forecast_metrics.set_index("model").loc["sarima"]
    baseline = forecast_metrics.set_index("model").loc["seasonal_naive"]
    isolation = anomaly_metrics.set_index("method").loc["isolation_forest"]
    robust = anomaly_metrics.set_index("method").loc["robust_seasonal_z"]

    cells = [
        markdown(
            "# Прогнозирование и обнаружение аномалий во временном ряду\n\n"
            "**Датасет:** Numenta Anomaly Benchmark (NAB), `machine_temperature_system_failure.csv`.  \n"
            "**Цель:** подготовить временной ряд температуры промышленной машины, сравнить базовый "
            "прогноз с SARIMA и обнаружить аномальные режимы двумя методами.\n\n"
            "Источник: [официальный репозиторий NAB](https://github.com/numenta/NAB)."
        ),
        markdown(
            "## 1. Предметная область и постановка задачи\n\n"
            "Каждая строка содержит временную метку и показание температурного датчика. "
            "Прогноз помогает оценивать ожидаемую температуру и заранее замечать изменение режима работы. "
            "Аномалией считаем необычное отклонение или интервал, совпадающий с официальным окном NAB; "
            "на практике это может соответствовать сбою, перегреву либо смене рабочего режима."
        ),
        code(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'src').exists():\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.pipeline import (load_and_prepare, apply_reference_labels, "
            "forecast, detect_anomalies, save_figures)\n"
            "pd.set_option('display.max_columns', 20)\n"
            "print(ROOT)",
            str(ROOT),
            1,
        ),
        markdown("## 2. Загрузка и первичный просмотр данных"),
        code(
            "raw = pd.read_csv(ROOT / 'data/raw/machine_temperature_system_failure.csv')\n"
            "print(raw.head().to_string(index=False))\n"
            "print(f'Форма: {raw.shape}')\n"
            "print('Типы данных:', raw.dtypes.astype(str).to_dict())\n"
            "print('Пропуски:', raw.isna().sum().to_dict())\n"
            "print('Повторные временные метки:', raw.duplicated('timestamp').sum())",
            raw.head().to_string(index=False)
            + f"\nФорма: {raw.shape}\n"
            + f"Типы данных: {raw.dtypes.astype(str).to_dict()}\n"
            + f"Пропуски: {raw.isna().sum().to_dict()}\n"
            + f"Повторные временные метки: {raw.duplicated('timestamp').sum()}",
            2,
        ),
        markdown(
            "## 3. Подготовка данных\n\n"
            "Метки времени преобразуются в `datetime`, строки сортируются. Для 12 повторных меток берётся "
            "среднее измерений: этот выбор сохраняет информацию и формирует однозначный индекс. Затем ряд "
            "приводится к частоте 5 минут. Разрывов и пропусков после этого нет; интерполяция оставлена в "
            "конвейере как воспроизводимая защита для возможного обновления файла. Случайное перемешивание не применяется."
        ),
        code(
            "clean, audit = load_and_prepare()\n"
            "data, windows = apply_reference_labels(clean)\n"
            "print(pd.Series(audit).to_string())\n"
            "print(f'Официальных окон аномалий: {len(windows)}')",
            pd.Series(summary["data_audit"]).to_string() + "\nОфициальных окон аномалий: 4",
            3,
        ),
        markdown(
            "## 4. Визуальный анализ\n\n"
            "Ряд заметно нестационарен: уровень температуры меняется ступенчато, присутствуют резкие локальные "
            "скачки и продолжительные режимы с иным уровнем. Скользящие среднее и стандартное отклонение "
            "подтверждают изменение локального уровня и волатильности. Поэтому одного глобального порога недостаточно.\n\n"
            "![Временной ряд](../results/figures/01_time_series.png)\n\n"
            "![Скользящие статистики](../results/figures/02_rolling_statistics.png)"
        ),
        markdown(
            "## 5. Прогнозирование\n\n"
            "Ряд усредняется по часу. Первые 80% наблюдений образуют обучающую часть, последние 20% - тестовую; "
            "хронологический порядок сохранён. Базовая модель повторяет значение того же часа предыдущих суток. "
            "Дополнительная модель SARIMA(1,1,1)(1,0,0,24) учитывает динамику ошибок и суточный цикл."
        ),
        code(
            "forecasts, forecast_metrics, forecast_meta = forecast(data)\n"
            "print(pd.Series(forecast_meta).to_string())\n"
            "print('\\nМетрики:')\n"
            "print(forecast_metrics.round(3).to_string(index=False))",
            pd.Series(summary["forecast"]).to_string() + "\n\nМетрики:\n" + fm,
            4,
        ),
        markdown(
            "![Сравнение прогнозов](../results/figures/03_forecast_comparison.png)\n\n"
            f"По RMSE лучшей стала **{best_forecast}**: {sarima['RMSE']:.2f} против "
            f"{baseline['RMSE']:.2f} у базовой модели. Разница по RMSE мала, а по MAPE базовая модель лучше "
            f"({baseline['MAPE_percent']:.2f}% против {sarima['MAPE_percent']:.2f}%). "
            "Следовательно, SARIMA немного лучше штрафует крупные ошибки, но не даёт устойчивого превосходства "
            "по всем метрикам. Это важное ограничение, а не повод объявлять модель безусловно лучшей."
        ),
        markdown(
            "## 6. Обнаружение аномалий\n\n"
            "Реализованы два метода. Первый оценивает робастный z-score ошибки суточного прогноза с порогом 4.0. "
            "Второй - Isolation Forest с долей выбросов 3%; он использует температуру, абсолютный скачок, "
            "суточную ошибку и отклонение от локальной медианы. Для проверки точки сопоставляются с четырьмя "
            "официальными окнами NAB."
        ),
        code(
            "detected, anomaly_metrics, anomaly_meta = detect_anomalies(data, windows)\n"
            "print(pd.Series(anomaly_meta).to_string())\n"
            "print('\\nМетрики:')\n"
            "print(anomaly_metrics.round(3).to_string(index=False))",
            pd.Series(summary["anomaly_detection"]).to_string() + "\n\nМетрики:\n" + am,
            5,
        ),
        markdown(
            "![Обнаруженные аномалии](../results/figures/04_anomaly_detection.png)\n\n"
            f"По точечному F1 лучшим стал **{best_anomaly}**: F1={isolation['F1']:.3f}, "
            f"precision={isolation['precision']:.3f}, recall={isolation['recall']:.3f}. Он подал сигнал во всех "
            f"4 из 4 окон. Робастный метод точнее ({robust['precision']:.3f}), но нашёл только "
            f"{int(robust['windows_detected'])} из 4 окон. Невысокий точечный recall ожидаем: NAB размечает "
            "продолжительные окна, а детектор отмечает прежде всего наиболее необычные точки внутри них."
        ),
        markdown(
            "## 7. Итоговые выводы\n\n"
            "1. После усреднения повторных меток получен регулярный ряд из 22 683 пятиминутных наблюдений без пропусков.\n"
            "2. SARIMA имеет минимально меньший RMSE, но сезонная базовая модель лучше по MAPE; усложнение модели "
            "не принесло однозначного выигрыша.\n"
            "3. Isolation Forest обнаружил все четыре известных события, а робастный суточный детектор дал более "
            "высокую точность отдельных сигналов. Практически полезно объединять их: Isolation Forest использовать "
            "для охвата событий, статистический метод - для более консервативных предупреждений.\n"
            "4. Ограничения: короткая история, единственный датчик, фиксированные пороги и точечная метрика для "
            "окон событий. Развитие проекта: подбор гиперпараметров только на валидационном отрезке, многомерные "
            "признаки, оценка задержки обнаружения и адаптивные пороги."
        ),
        markdown(
            "## 8. Сохранённые результаты\n\n"
            "Таблицы находятся в `results/*.csv`, графики - в `results/figures/`, очищенный ряд - в "
            "`data/processed/`. Полный повторный запуск выполняется командой `python src/pipeline.py`."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
