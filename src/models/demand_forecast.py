"""Skill demand forecasting with Prophet (default) or ARIMA."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.config import ROOT, load_config

warnings.filterwarnings("ignore", category=FutureWarning)


class DemandForecaster:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.method = self.config["models"]["forecast"]["method"]
        self.horizon = self.config["models"]["forecast"]["horizon_weeks"]
        self.forecasts: dict[str, pd.DataFrame] = {}
        self._arima_models: dict = {}

    def _prep_prophet_df(self, series: pd.DataFrame) -> pd.DataFrame:
        return series.rename(columns={"week_start": "ds", "demand_fte": "y"})[
            ["ds", "y"]
        ]

    def _forecast_prophet(self, history: pd.DataFrame) -> pd.DataFrame:
        from prophet import Prophet

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        train = self._prep_prophet_df(history)
        m.fit(train)
        future = m.make_future_dataframe(periods=self.horizon, freq="W-MON")
        fc = m.predict(future)
        result = fc.tail(self.horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        result = result.rename(
            columns={
                "ds": "week_start",
                "yhat": "forecast_fte",
                "yhat_lower": "forecast_lower",
                "yhat_upper": "forecast_upper",
            }
        )
        return result

    def _forecast_arima(self, history: pd.DataFrame, skill: str) -> pd.DataFrame:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        ts = history.set_index("week_start")["demand_fte"].astype(float)
        model = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 13),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        self._arima_models[skill] = fit
        pred = fit.get_forecast(steps=self.horizon)
        mean = pred.predicted_mean
        conf = pred.conf_int()
        last_date = ts.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=self.horizon,
            freq="W-MON",
        )
        return pd.DataFrame(
            {
                "week_start": future_dates,
                "forecast_fte": mean.values,
                "forecast_lower": conf.iloc[:, 0].values,
                "forecast_upper": conf.iloc[:, 1].values,
            }
        )

    def fit_predict(self, skill_demand: pd.DataFrame) -> pd.DataFrame:
        all_fc = []
        for skill, grp in skill_demand.groupby("skill"):
            hist = grp.sort_values("week_start").copy()
            # Use only historical portion (exclude extra future weeks if any)
            if self.method == "prophet":
                fc = self._forecast_prophet(hist)
            else:
                fc = self._forecast_arima(hist, skill)
            fc["skill"] = skill
            self.forecasts[skill] = fc
            all_fc.append(fc)

        combined = pd.concat(all_fc, ignore_index=True)
        return combined

    def get_skill_summary(self) -> pd.DataFrame:
        """Next-week demand vs current bench supply proxy."""
        rows = []
        for skill, fc in self.forecasts.items():
            next_w = fc.iloc[0]
            rows.append(
                {
                    "skill": skill,
                    "next_week_demand_fte": round(next_w["forecast_fte"], 2),
                    "demand_trend_8w": round(
                        fc["forecast_fte"].mean() - fc["forecast_fte"].iloc[0], 2
                    ),
                    "forecast_uncertainty": round(
                        next_w["forecast_upper"] - next_w["forecast_lower"], 2
                    ),
                }
            )
        return pd.DataFrame(rows)

    def save(self, directory: Path | None = None) -> None:
        d = directory or ROOT / self.config["paths"]["models_dir"]
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"forecasts": self.forecasts, "method": self.method, "horizon": self.horizon},
            d / "demand_forecaster.joblib",
        )

    def load(self, directory: Path | None = None) -> None:
        d = directory or ROOT / self.config["paths"]["models_dir"]
        payload = joblib.load(d / "demand_forecaster.joblib")
        self.forecasts = payload["forecasts"]
        self.method = payload["method"]
        self.horizon = payload["horizon"]
