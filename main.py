### IMPORT PACKAGES

import sqlite3
from enum import Enum
from fastapi import FastAPI, HTTPException, status
from typing import Dict, List, Literal, Optional
from datetime import date, datetime
from inspect import cleandoc

from pydantic import BaseModel, Field
from arch.unitroot import ADF
from statsmodels.stats.diagnostic import het_arch

from data import get_start_date, TwelveDataAPI, get_current_timestamp
from model import build_model


### DATA MODELS

class IdentifierType(str, Enum):
    ISIN = "isin"
    TICKER = "ticker"

class WindowPeriod(str, Enum):
    ONE_YEAR = "1y"
    THREE_YEAR = "3y"
    FIVE_YEAR = "5y"
    
class ErrorDistribution(str, Enum):
    NORMAL = "normal"
    STUDENTS_T = "studentst"
    SKEWT = "skewt"
    GED = "ged"
    
class Parameters(BaseModel):
    p: int = Field(default=1, description="Parameter p")
    q: int = Field(default=1, description="Parameter q")
    
class GARCHParameters(BaseModel):
    p: int = Field(default=1, ge=1, le=5, description="GARCH lag order")
    q: int = Field(default=1, ge=1, le=5, description="ARCH lag order")
    
class DataMetadata(BaseModel):
    identifier: str
    start_date: date
    end_date: date
    total_observations: int
    
class FitQualityMetrics(BaseModel):
    log_likelihood: float
    aic: float = Field(..., description="Akaike Information Criterion")
    bic: float = Field(..., description="Bayesian Information Criterion")
    converged: bool = Field(..., description="Whether the optimizer reached convergence")

class ParameterEstimate(BaseModel):
    value: float
    std_err: Optional[float] = None
    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    
class FitRequest(BaseModel):
    identifier: str = Field(
        ...,
        examples=["AAPL", "US0378331005"],
        description="Asset identifier (Ticker or ISIN)",
    )
    type: IdentifierType = Field(
        default=IdentifierType.TICKER,
        description="Type of the identifier provided",
    )
    window_period: WindowPeriod = Field(
        default=WindowPeriod.THREE_YEAR,
        description="Historical window used for fetching price data and fitting",
    )
    parameters: GARCHParameters = Field(
        default_factory=GARCHParameters,
        description="Lag orders for the GARCH(p, q) process",
    )
    distribution: ErrorDistribution = Field(
        default=ErrorDistribution.STUDENTS_T,
        description="Assumed residual error distribution",
    )
    include_coefficients: bool = Field(
        default=False,
        description="Whether to include estimated parameter coefficients in the response",
    )
    
class FitSummary(BaseModel):
    model: str
    parameters: GARCHParameters
    distribution: ErrorDistribution
    trained_at: datetime

class FitResponse(BaseModel):
    status: Literal["success", "failed"] = "success"
    name: str = Field(
        ...,
        examples=["20260904_140536-ARCH11_AAPL"],
        description="Artifact identifier formatted as {timestamp}-ARCH{p}{q}_{ticker}",
    )
    summary: FitSummary
    data: DataMetadata
    metrics: FitQualityMetrics
    coefficients: Optional[Dict[str, ParameterEstimate]] = Field(
        default=None,
        description="Estimated parameters; omitted when include_coefficients is false",
    )
  
class ErrorDetail(BaseModel):
    detail: str = Field(..., example="Optimizer failed to converge after 500 iterations.")

# Start FastAPI application
app = FastAPI(title="Financial Econometrics API")

# `"/hello" path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    """Return dictionary with greeting message."""
    return {"message" : "Hello, World"}

@app.get("/diagnostics/check", status_code=200)
def diagnostics(identifier : str, type : IdentifierType = 'ticker', period : WindowPeriod = '3y'):
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
    response_model=FitResponse, 
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
def fit(payload: FitRequest):
    """Fit a GARCH(p, q) volatility model for a specified financial asset.

    Fetches historical daily returns for the given asset identifier across
    the requested lookback window, estimates model parameters using maximum
    likelihood estimation, and returns goodness-of-fit metrics along with
    optional parameter estimates."""
    
    identifier = payload.identifier
    period = payload.window_period
    p,q = payload.parameters.p, payload.parameters.q
    dist = payload.distribution.value
    
    #Build GARCH Model
    model = build_model(identifier)
    model.get_daily_returns(period)
    model.fit(p,q,dist)
    
    model_name, path = model.dump()
    
    res = model.model
    
    coefficients_dict=None
    if payload.include_coefficients:
        coefficients_dict = {
            str(param): ParameterEstimate(
                value=res.params[param],
                std_err=getattr(res, "std_err", {}).get(param),
                t_stat=getattr(res, "tvalues", {}).get(param),
                p_value=getattr(res, "pvalues", {}).get(param),
            )
            for param in res.params.index
        }
    
    response = FitResponse(
        status = 'success',
        name = model_name,
        summary= FitSummary(
            model = 'GARCH',
            parameters = GARCHParameters(
                p = payload.parameters.p,
                q = payload.parameters.q
            ),
            distribution = payload.distribution.value,
            trained_at = model.trained_date
        ),
        data = DataMetadata(
            identifier=identifier,
            start_date=model.data.index[0],
            end_date=model.data.index[-1],
            total_observations=len(model.data)
        ),
        metrics=FitQualityMetrics(
                    log_likelihood=float(res.loglikelihood),
                    aic=float(res.aic),
                    bic=float(res.bic),
                    converged=bool(res.convergence_flag == 0),
                ),
        coefficients=coefficients_dict
    )
    return response

@app.post(
    "/models/forecast", 
    status_code=200
    )
def forecast():
    return {"message" : "This endpoint returns volatility and value-at-risk predictions."}

# @app.get("/models", status_code=200)
# def list_models():
#     return {"message" : "This endpoint returns a list of saved models based on a filter criteria."}

# @app.get("/models/{model_id}", status_code=200)
# def get_model():
#     return {"message" : "This endpoint returns key information about a saved model."}