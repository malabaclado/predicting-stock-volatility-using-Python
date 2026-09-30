### IMPORT PACKAGES

from fastapi import FastAPI, HTTPException, status
from inspect import cleandoc
import pandas as pd
import schemas as scm

from arch.unitroot import ADF
from statsmodels.stats.diagnostic import het_arch

from math_helper import calculate_risk_metrics_for_horizon, get_volatility_summary, get_horizon_forecasts
from data import get_start_date, TwelveDataAPI
from model import build_model, read_models_table, filter_saved_models, save_model_to_db


# ==========================================
# Math Functions
# ==========================================

# Start FastAPI application
app = FastAPI(title="Financial Econometrics API")

@app.get("/diagnostics/check", status_code=200)
def diagnostics(ticker: str, period: scm.WindowPeriod = scm.WindowPeriod.THREE_YEAR):
    """Tests stationarity and ARCH effects on a time-series of equity returns during a specified period."""
    ticker_clean = ticker.strip().upper()
    response = {
        "ticker": ticker_clean,
    }

    # Get start date
    period_val = period.value if hasattr(period, "value") else str(period)
    date_start = get_start_date(period_val)
    
    # Get data from API
    api = TwelveDataAPI()
    prices = api.fetch_data_from_api(ticker_clean, date_start)
    
    # Calculate % returns
    returns = 100 * prices['close'].pct_change().dropna()

    # 1. Check Stationarity
    adf = ADF(returns)

    # 2. Check for ARCH Effects (Engle's LM Test)
    # het_arch() returns: (lm_stat, p_value, f_stat, f_pvalue)
    lm_stat, p_value, _, _ = het_arch(returns, nlags=5)

    data = {
        "is_stationary": bool(adf.pvalue < 0.05),
        "has_arch_effects": bool(p_value < 0.05),
        "adf_pvalue": float(adf.pvalue),
        "arch_lm_pvalue": float(p_value),
        "recommendation": (
            "Proceed with GARCH(1,1)"
            if (adf.pvalue < 0.05 and p_value < 0.05)
            else "GARCH(1,1) may not be suitable for this series."
        ),
    }
    # Update response
    response['date'] = {
        "start": prices.index[-1].strftime("%Y-%m-%d"),
        "end": prices.index[0].strftime("%Y-%m-%d")
    }
    
    response['data'] = data
    
    return response


@app.post(
    "/model/search", 
    status_code=200,
    response_model_exclude_none=True,
)
def search_model(payload: scm.ModelSearchRequest):
    # Unpack payload
    converged = payload.converged

    # Read models database
    try:
        df = read_models_table()
    except Exception as e:
        return {"error": f"Error reading models table: {e}"}
    
    # Filter saved models df
    total, filtered_df = filter_saved_models(df, payload)
    
    saved_models = []
    
    for i in filtered_df.index:
        row = filtered_df.loc[i]
        model_spec = scm.ModelSpecsWithMetrics(
            model_name=i,
            model_type="GARCH",
            order=scm.GARCHOrder(
                p=int(row['order_p']),
                q=int(row['order_q'])
            ),
            distribution=row['distribution'],
            trained_at=row['trained_at'],
            data=scm.DataMetadata(
                ticker=row['ticker'],
                start_date=row["start_date"],
                end_date=row["end_date"],
                total_observations=int(row["total_observations"]),
                window_period=row["window_period"]
            ),
            metrics=scm.ModelMetrics(
                aic=round(float(row["aic"]), 4),
                bic=round(float(row["bic"]), 4),
                persistence=round(float(row["persistence"]), 4),
                converged=(int(row["converged"]) == 0)
            )
        )
        saved_models.append(model_spec)

    return scm.ModelSearchResponse(search_results=saved_models, total=total)


@app.post(
    "/models/fit", 
    response_model=scm.FitResponse, 
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED, 
    operation_id="fit_garch_model",
    summary="Fit GARCH(p,q) volatility model",
    description="""Fits an **autoregressive conditional heteroskedasticity (GARCH)** model on historical daily log returns.""",
    responses={
        201: {
            "description": "Model fitted successfully and artifact generated.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "name": "20260904-GARCH11_AAPL",
                        "data_summary": {
                            "ticker": "AAPL",
                            "start_date": "2023-09-01",
                            "end_date": "2026-09-01",
                            "total_observations": 754
                        },
                        "metrics": {
                            "log_likelihood": 2245.81,
                            "aic": -4481.62,
                            "bic": -4458.5,
                            "converged": True
                        },
                        "coefficients": {
                            "mu": {"value": 0.00084, "std_err": 0.00038, "t_stat": 2.21, "p_value": 0.027},
                            "omega": {"value": 0.000012, "std_err": 0.000004, "t_stat": 3.0, "p_value": 0.0027},
                            "alpha[1]": {"value": 0.085, "std_err": 0.019, "t_stat": 4.47, "p_value": 0.00001},
                            "beta[1]": {"value": 0.865, "std_err": 0.028, "t_stat": 30.89, "p_value": 0.0}
                        }
                    }
                }
            },
        },
    }
)
def fit(payload: scm.FitRequest):
    """Fit a GARCH(p, q) volatility model for a specified financial asset.

    Fetches historical daily returns for the given asset ticker symbol across
    the requested lookback window, estimates model parameters using maximum
    likelihood estimation, and returns goodness-of-fit metrics along with
    optional parameter estimates."""
    
    ticker = payload.ticker.strip().upper()
    period = payload.window_period
    period_val = period.value if hasattr(period, "value") else str(period)
    p, q = payload.parameters.p, payload.parameters.q
    dist = payload.distribution.value if hasattr(payload.distribution, "value") else str(payload.distribution)
    use_new_data = payload.use_new_data
    
    # Build GARCH Model
    model = build_model(ticker)
    model.get_daily_returns(period_val, use_new_data)
    model.fit(p, q, dist)
    
    # GARCH model results
    res = model.model
    
    coefficients_dict = None
    if payload.include_coefficients:
        coefficients_dict = {
            str(param): scm.ParameterEstimate(
                value=round(res.params[param], 4),
                std_err=round(getattr(res, "std_err", {}).get(param), 4),
                t_stat=round(getattr(res, "tvalues", {}).get(param), 4),
                p_value=round(getattr(res, "pvalues", {}).get(param), 4),
            )
            for param in res.params.index
        }
        
    # Build model artifacts for saving
    artifact = {
        "model_result": res,                           # The ARCHModelResult object
        "identifier": ticker,
        "ticker": ticker,
        "p": p,                                         # GARCH lag p
        "q": q,                                         # GARCH lag q
        "distribution": dist,                           # e.g., "studentst"
        "trained_at": model.trained_date,               # Training timestamp
        "last_price_date": model.data.index[-1],        # Cutoff date from data index
        "data_summary": {
            "start_date": model.data.index[0],
            "end_date": model.data.index[-1],
            "window_period": period_val,
            "total_observations": len(model.data)
        }
    }
    
    # Save model artifact (.pkl file)
    model_name = model.dump(artifact)
    
    # Save metadata to SQLite models table
    alpha_sum = sum(val for key, val in res.params.items() if key.startswith("alpha"))
    beta_sum = sum(val for key, val in res.params.items() if key.startswith("beta"))
    persistence = alpha_sum + beta_sum

    model_record = {
        "model_name": model_name,
        "ticker": ticker,
        "distribution": str(dist),
        "trained_at": str(model.trained_date),
        "start_date": str(model.data.index[0]),
        "end_date": str(model.data.index[-1]),
        "window_period": str(period_val),
        "total_observations": int(len(model.data)),
        "order_p": int(p),
        "order_q": int(q),
        "converged": int(res.convergence_flag) if hasattr(res, "convergence_flag") else 0,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "persistence": round(float(persistence), 4),
    }
    save_model_to_db(model_record)
    
    # Build response payload
    response = scm.FitResponse(
        status='success',
        name=model_name,
        summary=scm.FitSummary(
            model='GARCH',
            parameters=scm.GARCHOrder(
                p=payload.parameters.p,
                q=payload.parameters.q
            ),
            distribution=dist,
            trained_at=model.trained_date
        ),
        data=scm.DataMetadata(
            ticker=ticker,
            start_date=model.data.index[0],
            end_date=model.data.index[-1],
            total_observations=len(model.data),
            window_period=period_val
        ),
        metrics=scm.FitQualityMetrics(
            log_likelihood=round(float(res.loglikelihood), 4),
            aic=round(float(res.aic), 4),
            bic=round(float(res.bic), 4),
            converged=bool(res.convergence_flag == 0),
        ),
        coefficients=coefficients_dict
    )
    
    return response


@app.post(
    "/models/forecast", 
    response_model=scm.ForecastResponse,
    response_model_exclude_none=True,
    status_code=200
)
def forecast(payload: scm.ForecastRequest):
    ticker = payload.ticker.strip().upper()
    model_name = payload.use_model
    annualization_factor = payload.annualization_factor
    confidence_levels = payload.value_at_risk.confidence_levels
    portfolio_value = payload.portfolio_value
    
    if payload.horizon:
        max_horizon = max(payload.horizon)
    else:
        max_horizon = 1
    
    # Load saved model
    try:
        model = build_model(ticker)
        artifact = model.load(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    
    res = artifact['model_result']
    dist_name = artifact['distribution']
    last_price_date = artifact['last_price_date']
    data_summary = artifact['data_summary']
    
    # Forecast daily conditional volatility
    forecasts = res.forecast(horizon=max_horizon, reindex=False)
    daily_variances = forecasts.variance.values[-1]

    volatility_summary = get_volatility_summary(res, annualization_factor)
    risk_summary = calculate_risk_metrics_for_horizon(
        volatility_summary.conditional_next_day, 
        confidence_levels, 
        portfolio_value,
        dist_name,
        res.params
    )
    
    horizon_forecasts = get_horizon_forecasts(payload, res, daily_variances, dist_name, last_price_date) if payload.horizon else None
    
    start_dt = data_summary['start_date'].date() if hasattr(data_summary['start_date'], 'date') else pd.to_datetime(data_summary['start_date']).date()
    end_dt = data_summary['end_date'].date() if hasattr(data_summary['end_date'], 'date') else pd.to_datetime(data_summary['end_date']).date()

    response = scm.ForecastResponse(
        status='success',
        portfolio_value=portfolio_value,
        summary=scm.ForecastSummary(
            volatility=volatility_summary,
            risk=risk_summary
        ),
        horizon_forecasts=horizon_forecasts,
        model_spec=scm.ModelSpec(
            model_name=artifact['model_name'],
            model_type="GARCH",
            order=scm.GARCHOrder(
                p=artifact['p'],
                q=artifact['q']
            ),
            distribution=dist_name,
            trained_at=artifact['trained_at'],
            data=scm.DataMetadata(
                ticker=ticker,
                start_date=start_dt,
                end_date=end_dt,
                total_observations=data_summary['total_observations'],
                window_period=data_summary['window_period']
            )
        )
    )
    return response

# @app.get("/models", status_code=200)
# def list_models():
#     return {"message" : "This endpoint returns a list of saved models based on a filter criteria."}

# @app.get("/models/{model_id}", status_code=200)
# def get_model():
#     return {"message" : "This endpoint returns key information about a saved model."}
