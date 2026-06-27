# Данные

В проекте используется временной ряд температуры промышленной машины из Numenta Anomaly Benchmark (NAB).

- Исходный файл: `raw/machine_temperature_system_failure.csv`
- Официальная разметка: `raw/combined_windows.json`
- Очищенный ряд: `processed/machine_temperature_clean.csv`
- Частота наблюдений: 5 минут
- Период: 2013-12-02 21:15:00 - 2014-02-19 15:25:00

Источник: https://github.com/numenta/NAB

Прямые адреса исходных файлов:

- https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv
- https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json

При подготовке 12 повторяющихся временных меток объединены средним значением. После приведения к регулярной сетке пропусков нет. В обработанном файле дополнительно сохранены эталонная разметка, признаки и сигналы обоих методов обнаружения.
