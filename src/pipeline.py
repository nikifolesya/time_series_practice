from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data/raw/machine_temperature_system_failure.csv"
LABELS = ROOT / "data/raw/combined_windows.json"
PROCESSED = ROOT / "data/processed/machine_temperature_clean.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SERIES_KEY = "realKnownCause/machine_temperature_system_failure.csv"


def configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def load_and_prepare() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(RAW_DATA)
    audit = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "missing_values": int(raw.isna().sum().sum()),
        "duplicate_timestamps": int(raw.duplicated("timestamp").sum()),
    }
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Повторные измерения в одну и ту же минуту усредняются: ни одно наблюдение
    # не отбрасывается, а временной индекс становится однозначным.
    clean = raw.groupby("timestamp", as_index=True)["value"].mean().to_frame()
    clean = clean.asfreq("5min")
    missing_after_reindex = int(clean["value"].isna().sum())
    clean["value"] = clean["value"].interpolate(method="time").ffill().bfill()
    clean.index.name = "timestamp"
    audit.update(
        {
            "clean_rows": int(len(clean)),
            "missing_after_reindex": missing_after_reindex,
            "remaining_missing": int(clean["value"].isna().sum()),
            "start": str(clean.index.min()),
            "end": str(clean.index.max()),
            "frequency": "5 minutes",
        }
    )
    return clean, audit


def apply_reference_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    with LABELS.open(encoding="utf-8") as file:
        windows = json.load(file)[SERIES_KEY]
    labeled = df.copy()
    labeled["is_anomaly_true"] = False
    parsed_windows = []
    for start, end in windows:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        labeled.loc[start_ts:end_ts, "is_anomaly_true"] = True
        parsed_windows.append((start_ts, end_ts))
    return labeled, parsed_windows


def forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Почасовое усреднение снижает шум и делает суточную сезонность (24 отсчета)
    # явной и вычислительно доступной для учебной SARIMA.
    hourly = df["value"].resample("1h").mean().dropna()
    split = int(len(hourly) * 0.8)
    train, test = hourly.iloc[:split], hourly.iloc[split:]

    seasonal_naive = hourly.shift(24).reindex(test.index)
    if seasonal_naive.isna().any():
        seasonal_naive = pd.Series(train.iloc[-1], index=test.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=120)
    sarima = fitted.get_forecast(steps=len(test))
    sarima_mean = sarima.predicted_mean
    conf = sarima.conf_int(alpha=0.05)

    forecasts = pd.DataFrame(
        {
            "actual": test,
            "seasonal_naive": seasonal_naive,
            "sarima": sarima_mean,
            "sarima_lower_95": conf.iloc[:, 0].to_numpy(),
            "sarima_upper_95": conf.iloc[:, 1].to_numpy(),
        }
    )

    rows = []
    for name in ["seasonal_naive", "sarima"]:
        y_true, y_pred = forecasts["actual"], forecasts[name]
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "MAPE_percent": mean_absolute_percentage_error(y_true, y_pred) * 100,
            }
        )
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metadata = {
        "hourly_observations": int(len(hourly)),
        "train_observations": int(len(train)),
        "test_observations": int(len(test)),
        "split_timestamp": str(test.index.min()),
        "sarima_aic": float(fitted.aic),
    }
    return forecasts, metrics, metadata


def detect_anomalies(
    df: pd.DataFrame, windows: list
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    detected = df.copy()
    value = detected["value"]

    # Метод 1: неожиданное отклонение от значения ровно сутки назад.
    seasonal_error = value - value.shift(288)
    center = seasonal_error.rolling(2016, min_periods=288).median()
    mad = (seasonal_error - center).abs().rolling(2016, min_periods=288).median()
    robust_z = 0.6745 * (seasonal_error - center) / mad.replace(0, np.nan)
    detected["seasonal_error"] = seasonal_error
    detected["robust_z"] = robust_z
    detected["anomaly_robust_z"] = robust_z.abs().gt(4.0).fillna(False)

    # Метод 2: Isolation Forest учитывает уровень, скачок, суточную ошибку и
    # отклонение от локальной медианы. Пропуски возникают только в начале ряда.
    features = pd.DataFrame(index=detected.index)
    features["value"] = value
    features["abs_diff"] = value.diff().abs()
    features["abs_daily_error"] = seasonal_error.abs()
    features["local_deviation"] = (
        value - value.rolling(288, min_periods=24).median()
    ).abs()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    forest = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )
    detected["anomaly_isolation_forest"] = forest.fit_predict(scaled) == -1
    detected["isolation_score"] = -forest.score_samples(scaled)

    metrics_rows = []
    y_true = detected["is_anomaly_true"].astype(bool)
    for method, column in [
        ("robust_seasonal_z", "anomaly_robust_z"),
        ("isolation_forest", "anomaly_isolation_forest"),
    ]:
        y_pred = detected[column].astype(bool)
        windows_found = sum(bool(y_pred.loc[start:end].any()) for start, end in windows)
        metrics_rows.append(
            {
                "method": method,
                "anomalous_points": int(y_pred.sum()),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "windows_detected": int(windows_found),
                "windows_total": int(len(windows)),
            }
        )
    metrics = pd.DataFrame(metrics_rows).sort_values("F1", ascending=False)
    metadata = {
        "true_anomalous_points": int(y_true.sum()),
        "true_anomaly_share_percent": float(y_true.mean() * 100),
        "reference_windows": int(len(windows)),
    }
    return detected, metrics, metadata


def save_figures(
    df: pd.DataFrame,
    forecasts: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    anomaly_metrics: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df.index, df["value"], color="#2563eb", linewidth=0.65)
    ax.set(title="Температура промышленной машины", xlabel="Время", ylabel="Температура")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_time_series.png", bbox_inches="tight")
    plt.close(fig)

    rolling = df["value"].rolling(288, min_periods=1)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(df.index, df["value"], color="#94a3b8", linewidth=0.45, label="Наблюдение")
    ax.plot(df.index, rolling.mean(), color="#dc2626", linewidth=1.2, label="Среднее за 24 часа")
    ax.plot(df.index, rolling.std(), color="#16a34a", linewidth=0.9, label="Ст. отклонение за 24 часа")
    ax.set(title="Скользящие статистики", xlabel="Время", ylabel="Значение")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_rolling_statistics.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(forecasts.index, forecasts["actual"], color="#111827", linewidth=1.2, label="Факт")
    ax.plot(forecasts.index, forecasts["seasonal_naive"], color="#f59e0b", linewidth=0.9, alpha=0.85, label="Сезонная базовая")
    ax.plot(forecasts.index, forecasts["sarima"], color="#2563eb", linewidth=1.0, label="SARIMA")
    ax.fill_between(
        forecasts.index,
        forecasts["sarima_lower_95"].to_numpy(),
        forecasts["sarima_upper_95"].to_numpy(),
        color="#93c5fd",
        alpha=0.22,
        label="95% интервал SARIMA",
    )
    ax.set(title="Прогноз на тестовом интервале", xlabel="Время", ylabel="Температура")
    ax.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_forecast_comparison.png", bbox_inches="tight")
    plt.close(fig)

    methods = [
        ("anomaly_robust_z", "Робастная ошибка суточного прогноза", "#dc2626"),
        ("anomaly_isolation_forest", "Isolation Forest", "#7c3aed"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, (column, title, color) in zip(axes, methods):
        ax.plot(df.index, df["value"], color="#64748b", linewidth=0.55)
        true = df["is_anomaly_true"]
        ax.fill_between(df.index, df["value"].min(), df["value"].max(), where=true, color="#fbbf24", alpha=0.13, label="Окна NAB")
        points = df[column]
        ax.scatter(df.index[points], df.loc[points, "value"], s=8, color=color, label="Сигналы метода", zorder=3)
        ax.set_title(title)
        ax.set_ylabel("Температура")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Время")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_anomaly_detection.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    forecast_metrics.set_index("model")[["MAE", "RMSE"]].plot.bar(ax=axes[0], color=["#2563eb", "#f59e0b"])
    axes[0].set(title="Ошибки прогноза", xlabel="Модель", ylabel="Ошибка")
    axes[0].tick_params(axis="x", rotation=0)
    anomaly_metrics.set_index("method")[["precision", "recall", "F1"]].plot.bar(ax=axes[1], color=["#2563eb", "#16a34a", "#dc2626"])
    axes[1].set(title="Качество обнаружения", xlabel="Метод", ylabel="Доля", ylim=(0, 1))
    axes[1].tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_metrics.png", bbox_inches="tight")
    plt.close(fig)


def run_pipeline() -> dict:
    configure_plotting()
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)

    clean, audit = load_and_prepare()
    labeled, windows = apply_reference_labels(clean)
    forecasts, forecast_metrics, forecast_meta = forecast(labeled)
    detected, anomaly_metrics, anomaly_meta = detect_anomalies(labeled, windows)

    detected.to_csv(PROCESSED, index=True)
    forecasts.to_csv(RESULTS / "forecast_values.csv", index=True)
    forecast_metrics.to_csv(RESULTS / "forecast_metrics.csv", index=False)
    anomaly_metrics.to_csv(RESULTS / "anomaly_metrics.csv", index=False)
    detected.loc[
        detected["anomaly_robust_z"] | detected["anomaly_isolation_forest"],
        ["value", "is_anomaly_true", "anomaly_robust_z", "anomaly_isolation_forest"],
    ].to_csv(RESULTS / "detected_anomalies.csv", index=True)
    save_figures(detected, forecasts, forecast_metrics, anomaly_metrics)

    summary = {
        "data_audit": audit,
        "forecast": forecast_meta,
        "anomaly_detection": anomaly_meta,
        "best_forecast_model_by_rmse": forecast_metrics.iloc[0]["model"],
        "best_anomaly_method_by_f1": anomaly_metrics.iloc[0]["method"],
    }
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
