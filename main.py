### IMPORT PACKAGES

from fastapi import FastAPI, HTTPException, status
from inspect import cleandoc
import schemas as scm

from arch.unitroot import ADF
from statsmodels.stats.diagnostic import het_arch

from math_helper import calculate_risk_metrics_for_horizon, get_volatility_summary, get_horizon_forecasts
from data import get_start_date, TwelveDataAPI, get_current_timestamp
from model import build_model


# ==========================================
# Math Functions
# ==========================================




# Start FastAPI application
app = FastAPI(title="Financial Econometrics API")

# `"/hello" path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    """Return dictionary with greeting message."""
    return {"message" : "Hello, World"}

@app.get("/diagnostics/check", status_code=200)
def diagnostics(identifier : str, type : scm.IdentifierType = 'ticker', period : scm.WindowPeriod = '3y'):
    """Tests stationarity and ARCH effects on a time-series of equity returns during a specified period."""
    
    response = {
        "id" : identifier,
        "type" : type,
    }

    # Get start date
    date_start = get_start_date(period)
    
    # Get data from API
    api = TwelveDataAPI()
    prices = api.fetch_data_from_api(identifier, date_start, type)
    
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
            "start" : prices.index[-1].strftime("%Y-%m-%d"),
            "end" : prices.index[0].strftime("%Y-%m-%d")
        }
    
    response['data'] = data
    
    return response


@app.post(
    "/models/fit", 
    response_model=scm.FitResponse, 
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED, 
    operation_id="fit_garch_model",
    summary="Fit GARCH(p,q) volatility model",
    description="""Fits an **autoregressive conditional heteroskedasticity (GARCH)** model on historical daily log returns.""",
    responses = {
            201: {
                "description": "Model fitted successfully and artifact generated.",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "success",
                            "name": "20260904-GARCH11_AAPL",
                            "data_summary": {
                                "identifier": "AAPL",
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
            # 400: {
            #     "model": ErrorDetail,
            #     "description": "Insufficient observations or invalid parameter configuration."
            # },
            # 422: {
            #     "description": "Validation error: invalid request payload or unrecognized identifier format."
            # },
            # 500: {
            #     "model": ErrorDetail,
            #     "description": "Model optimization failure or internal data retrieval failure."
            # }
        }
    )
def fit(payload: scm.FitRequest):
    """Fit a GARCH(p, q) volatility model for a specified financial asset.

    Fetches historical daily returns for the given asset identifier across
    the requested lookback window, estimates model parameters using maximum
    likelihood estimation, and returns goodness-of-fit metrics along with
    optional parameter estimates."""
    
    identifier = payload.identifier
    period = payload.window_period
    p,q = payload.parameters.p, payload.parameters.q
    dist = payload.distribution.value
    use_new_data = payload.use_new_data
    
    #Build GARCH Model
    model = build_model(identifier)
    model.get_daily_returns(period, use_new_data)
    model.fit(p,q,dist)
    
    # GARCH model results
    res = model.model
    
    coefficients_dict=None
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
        
        
    # Build model artifacts for saviing
    artifact = {
                "model_result": res,                           # The ARCHModelResult object
                "identifier": identifier,                       # e.g., "AAPL"
                "p": p,                                         # GARCH lag p
                "q": q,                                         # GARCH lag q
                "distribution": dist,                           # e.g., "studentst"
                "trained_at": model.trained_date,                # Training timestamp
                "last_price_date": model.data.index[-1],   # Cutoff date from data index
                "data_summary": {
                    "start_date": model.data.index[0],
                    "end_date": model.data.index[-1],
                    "total_observations": len(model.data)
                }
            }
    
    # Save model 
    model_name = model.dump(artifact)
    
    # Build response payload
    response = scm.FitResponse(
        status = 'success',
        name = model_name,
        summary= scm.FitSummary(
            model = 'GARCH',
            parameters = scm.GARCHOrder(
                p = payload.parameters.p,
                q = payload.parameters.q
            ),
            distribution = payload.distribution.value,
            trained_at = model.trained_date
        ),
        data = scm.DataMetadata(
            identifier=identifier,
            start_date=model.data.index[0],
            end_date=model.data.index[-1],
            total_observations=len(model.data)
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
    ticker = payload.identifier
    model_name = payload.use_model
    annualization_factor = payload.annualization_factor
    confidence_levels = payload.value_at_risk.confidence_levels
    portfolio_value = payload.portfolio_value
    
    if payload.horizon:
        max_horizon = max(payload.horizon)
    else:
        max_horizon = 1
    
    # Load saved model
    model = build_model(ticker)
    artifact = model.load(model_name)
    
    res = artifact['model_result']
    dist_name = artifact['distribution']
    last_price_date = artifact['last_price_date']
    
    # Forecast daily conditional volatility
    forecasts = res.forecast(horizon=max_horizon, reindex=False)
    daily_variances = forecasts.variance.values[-1]

    volatility_summary = get_volatility_summary(res, annualization_factor)
    risk_summary = calculate_risk_metrics_for_horizon(volatility_summary.conditional_next_day, 
                                    confidence_levels, 
                                    portfolio_value,
                                    dist_name,
                                    res.params)
    
    horizon_forecasts = get_horizon_forecasts(payload, res, daily_variances, dist_name, last_price_date) if payload.horizon else None
    
    response = scm.ForecastResponse(
        status = 'success',
        portfolio_value=portfolio_value,
        summary=scm.ForecastSummary(
            volatility=volatility_summary,
            risk=risk_summary
        ),
        horizon_forecasts=horizon_forecasts,
        model_spec = scm.ModelSpec(
                    model_name = artifact['model_name'],
                    model_type = "GARCH",
                    order = scm.GARCHOrder(
                        p = artifact['p'],
                        q = artifact['q']
                    ),
                    distribution = dist_name,
                    trained_at = artifact['trained_at'],
                    data = scm.ModelSpecData(
                        identifier=ticker,
                        last_price_date=last_price_date
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