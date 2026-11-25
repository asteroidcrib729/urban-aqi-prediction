# Pakistan Air Quality & Winter Smog Forecasting

End-to-end exploratory data analysis, leak-free next-day (`t+1`) **PM2.5**
(`LinearRegression`) and **Winter Smog Episode** (`LogisticRegression`)
forecasting across **30 major Pakistani cities**, and an interactive city
monitoring dashboard unified in a single Jupyter notebook.

## Project Structure

- `scripts/fetch_pakistan_data.py` - Reproducible Open-Meteo data pipeline
  fetching hourly CAMS air quality + ERA5 historical weather across 30 Pakistani
  cities (`2022-08-05` to `2026-11-24`) and aggregating them into daily records
  with zero missing values.
- `urban-aqi-prediction.ipynb` - Unified end-to-end notebook covering:
  1. **Environment Setup & Data Loading** - Loads
     `dataset/pakistan_air_quality_daily.csv`, parses dates (`%Y-%m-%d`),
     encodes `city_id` (`0-29`), and derives meteorological inversion and
     calendar features (`temp_range`, `dew_point_spread`, `month`,
     `day_of_week`, `day_of_year`).
  2. **Exploratory Data Analysis & Winter Smog Seasonality** - Inspects CAMS
     pollutant and ERA5 weather correlations with `pm2_5_mean`, Pakistan PM2.5
     AQI category distribution, winter smog (`Nov-Feb`) vs. non-winter mean
     PM2.5 across top cities (`visualizations/eda_overview.png`), and
     single-city PACF (`Lahore`, `visualizations/lahore_pacf.png`).
  3. **City-Grouped Feature Engineering & Dual Modeling** - Builds `88`
     leak-free `city_id`-grouped lag (`1, 2, 3` days) and rolling (`3, 7` days)
     features, splits chronologically per city (`80/10/10`), trains
     `LinearRegression` for next-day `pm2_5_mean` (`t+1`) and a balanced
     `LogisticRegression` pipeline for next-day smog episodes
     (`PM2.5 >= 55.5 µg/m³`), evaluates overall, winter-smog, and per-city
     diagnostics (`visualizations/model_evaluation.png`,
     `visualizations/winter_smog_city_ranking.png`), and saves
     `model/pakistan_aqi_smog_models.pkl`.
  4. **Interactive Pakistan City & Winter Smog Dashboard (`ipywidgets` +
     `folium`)** - Displays next-day `PM2.5` forecast, winter smog risk
     probability (`%`), 30-day actual vs. predicted chart
     (`visualizations/city_dashboard_overview.png`), top-6 correlations, an
     all-city forecast table, and a 30-city Folium map (`Esri.WorldTopoMap`)
     exported to `visualizations/pakistan_air_quality_map.html`.
- `dataset/`
  - `pakistan_air_quality_daily.csv` - Unified daily air quality & ERA5 weather
    dataset (`47,190` rows x `29` columns, `2022-08-05` to `2026-11-24`, `30`
    Pakistani cities across all provinces/territories, `0` missing values),
    fetched from Open-Meteo ([CAMS Air Quality](https://open-meteo.com/en/docs/air-quality-api)
    and [ERA5 Historical Weather](https://open-meteo.com/en/docs/historical-weather-api)).
- `model/`
  - `pakistan_aqi_smog_models.pkl` - Serialized model bundle (`pm25_model`,
    `smog_classifier`, `feature_columns`, `city_map`, `smog_threshold_ugm3`).
- `visualizations/`
  - `eda_overview.png` - Pollutant/weather correlations, Pakistan PM2.5 AQI
    category distribution, and top 12 cities winter smog vs. non-winter PM2.5.
  - `lahore_pacf.png` - Single-city partial autocorrelation plots for
    `pm2_5_mean` and `pm10_mean` (`Lahore`).
  - `model_evaluation.png` - Validation/test actual vs. predicted scatter plots,
    residuals, and smog episode classifier ROC curves.
  - `winter_smog_city_ranking.png` - Per-city winter smog season (`Nov 2026`)
    actual vs. predicted mean PM2.5 and holdout test RMSE across all 30 cities.
  - `city_dashboard_overview.png` - Last 30-day actual vs. predicted trend and
    top-6 pollutant/weather correlations for the selected city (`Lahore` by
    default).
  - `pakistan_air_quality_map.html` - Interactive 30-city Folium map generated
    by Section 4 using keyless `Esri.WorldTopoMap` tiles for reliable local
    `file://` and browser rendering.
- `tests/`
  - `test_project.py` - Automated verification suite validating configs,
    notebook compilation/outputs, dataset invariants, serialized model metrics,
    exported charts, and map tile/popup regressions.
- `.github/workflows/ci.yml` - GitHub Actions workflow running Ruff, Pyrefly,
  and `unittest` on Python `3.14`.

## Verified Metrics Snapshot

- **Dataset:** `47,190` rows x `29` columns (`30` Pakistani cities,
  `2022-08-05` to `2026-11-24`, `0` missing values, `12,287` smog days /
  `26.0%`)
- **Winter Smog Seasonality:** Winter (`Nov-Feb`) mean PM2.5 `68.18` µg/m³ vs.
  Non-Winter (`Mar-Oct`) `36.37` µg/m³
- **Engineered Features:** `88` leak-free city-grouped pollutant + ERA5 weather
  features
- **Chronological Splits:** Train `37,560` | Validation `4,710` | Test `4,710`
- **Validation (`t+1`):** All `RMSE 14.24 | MAE 9.10 | R² 0.8076` | Winter
  `RMSE 20.38 | MAE 13.55` | Smog `ROC-AUC 0.9487 | PR-AUC 0.8461`
- **Test (`t+1`):** All `RMSE 13.35 | MAE 8.82 | R² 0.7723` | Winter
  `RMSE 22.05 | MAE 15.01` | Smog `ROC-AUC 0.9372 | PR-AUC 0.8598`

## Python & IDE Configuration

This project targets the global Python `3.14.7` interpreter without requiring a
project-local `.venv`.

- `.python-version` - Pins Python `3.14.7`.
- `pyproject.toml` - Declares `requires-python = ">=3.14.7,<3.15"`, runtime
  dependencies, optional typing stubs (`pandas-stubs`, `types-requests`,
  `types-seaborn`), MIT license metadata, and Ruff lint settings (`E4`, `E7`,
  `E9`, `F`, `I`).
- `requirements.txt` - Flexible version bounds (`>=installed,<next_major` for
  SemVer packages and `>=installed` for CalVer stubs).
- `pyrefly.toml` - Configures Pyrefly across `**/*.py*` and `**/*.ipynb` using
  Python `3.14` and the global/user `site-packages`.
- `.vscode/settings.json` & `.vscode/extensions.json` - Configures VS Code for
  system Python (`ms-python.python:system` + `pip`), bundled Pyrefly
  (`python.languageServer: "None"`, `python.pyrefly.syncNotebooks: true`),
  Ruff, Jupyter kernel filtering (`jupyter.excludeUserSitePackages: false`), and
  recommended extensions (`python`, `vscode-python-envs`, `pyrefly`, `jupyter`,
  `ruff`, `markdownlint`).

## First-Time Run Checklist

1. Verify Python `3.14.7` is active:

   ```bash
   python --version
   ```

2. Install runtime and typing dependencies into the global Python environment:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Refresh `dataset/pakistan_air_quality_daily.csv` from Open-Meteo:

   ```bash
   python scripts/fetch_pakistan_data.py
   ```

4. Run static checks and automated verification tests:

   ```bash
   pyrefly check
   python -m unittest discover -s tests -v
   ```

5. Open and run `urban-aqi-prediction.ipynb` in VS Code (select the Python
   `3.14.7` kernel) or via standalone Jupyter:

   ```bash
   pip install notebook
   jupyter notebook urban-aqi-prediction.ipynb
   ```

## Troubleshooting

- **Missing packages in Jupyter kernel:** Confirm VS Code selects the Python
  `3.14.7` interpreter configured in `.vscode/settings.json` and that
  `jupyter.excludeUserSitePackages` remains `false`.
- **Map tiles failing locally:** `visualizations/pakistan_air_quality_map.html`
  uses `Esri.WorldTopoMap` instead of OpenStreetMap/CARTO so tiles load cleanly
  over `file://` without `Referer` errors or API-key watermarks.
- **Pyrefly cannot find third-party packages:** Ensure `pyrefly.toml`
  `python-interpreter-path` and `site-package-path` match your local Python
  `3.14` installation paths (or pass `--python-interpreter-path` and
  `--site-packages-path` in CI).

## License

Licensed under the [MIT License](LICENSE).
