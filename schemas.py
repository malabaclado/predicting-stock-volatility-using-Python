from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Literal, Optional
from datetime import date, datetime

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
    
class GARCHOrder(BaseModel):
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
    
class ErrorDetail(BaseModel):
    detail: str = Field(..., example="Optimizer failed to converge after 500 iterations.")

# ==========================================
# Schemas for /model/fit
# ==========================================

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
    parameters: GARCHOrder = Field(
        default_factory=GARCHOrder,
        description="Lag orders for the GARCH(p, q) process",
    )
    distribution: ErrorDistribution = Field(
        default=ErrorDistribution.STUDENTS_T,
        description="Assumed residual error distribution",
    )
    use_new_data: bool = Field(
        default = False,
        examples=[False, True],
        description="Overrides the caching logic to force the API to use the latest data",
    )
    include_coefficients: bool = Field(
        default=False,
        description="Whether to include estimated parameter coefficients in the response",
    )
    
class FitSummary(BaseModel):
    model: str
    parameters: GARCHOrder
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
  
# ==========================================
# Schemas for /model/forecast
# ==========================================

class VaRConfig(BaseModel):
    confidence_levels: List[float] = Field(
        default=[0.95, 0.99],
        description="Confidence thresholds for VaR and ES (e.g. 0.95, 0.99).",
    )
    # method: Literal["parametric", "historical"] = Field(
    #     default="parametric",
    #     description="Estimation methodology for tail risk.",
    # )

    @field_validator("confidence_levels")
    @classmethod
    def validate_confidence_levels(cls, levels: List[float]) -> List[float]:
        if not levels:
            raise ValueError("confidence_levels list cannot be empty.")
        clean_levels = sorted(set(levels))
        for level in clean_levels:
            if not (0.5 < level < 1.0):
                raise ValueError(f"Confidence level {level} must be between 0.5 and 1.0.")
        return clean_levels

class ForecastRequest(BaseModel):
    identifier: str = Field(
            ...,
            examples=["AAPL", "US0378331005"],
            description="Asset identifier (Ticker or ISIN)",
        )
    type: IdentifierType = Field(
            default=IdentifierType.TICKER,
            description="Type of the identifier provided",
        )
    use_model: str = Field(
        default="latest",
        examples=["latest", "2026-09-05T20-03-05.076395-GARCH11_AAPL"],
        description="Use a specific model from saved models. Defaults to 'latest'.",
    )
    horizon: Optional[List[int]] = Field(
        default=None,
        examples=[[1,5,20,60]],
        description="Target forecast holding periods in trading days.",
    )
    portfolio_value: float = Field(
        default=10000.0,
        gt=0,
        examples=[1000000.00],
        description="Nominal position size in account base currency.",
    )
    annualization_factor: int = Field(
        default=252,
        ge=1,
        le=365,
        examples=[252],
        description="Annual trading days multiplier (e.g., 252 for equities, 365 for crypto).",
    )
    value_at_risk: VaRConfig = Field(
        default_factory=VaRConfig,
        description="Configuration parameters for VaR and Expected Shortfall calculations.",
    )

    @field_validator("horizon")
    @classmethod
    def validate_horizons(cls, horizons: List[int]) -> List[int]:
        if horizons is None:
            return None
        if not horizons:
            raise ValueError("Horizon list cannot be empty.")
        clean_horizons = sorted(set(horizons))
        if any(h < 1 for h in clean_horizons):
            raise ValueError("All horizon days must be >= 1.")
        if any(h > 252 for h in clean_horizons):
            raise ValueError("Horizon cannot exceed 252 trading days.")
        return clean_horizons


class ModelSpecData(BaseModel):
    identifier: str = Field(..., description="The asset identifier (e.g. ticker or ISIN)")
    last_price_date: date = Field(..., description="The cutoff date of the historical pricing data used")

class ModelSpec(BaseModel):
    model_name: str = Field(..., description="The unique name of the fitted model artifact used")
    model_type: Literal["GARCH"] = Field(default="GARCH", description="The class of volatility model")
    order: GARCHOrder = Field(..., description="Lag orders of the GARCH model")
    distribution: str = Field(..., description="The error distribution assumed by the model (e.g. normal, studentst)")
    trained_at: datetime = Field(..., description="Timestamp of when the model was trained")
    data: ModelSpecData = Field(..., description="Metadata about the underlying data used by the model")

class VolatilitySummary(BaseModel):
    conditional_next_day: float = Field(..., description="1-day ahead forecast daily conditional volatility (percentage units)")
    conditional_annualized: float = Field(..., description="1-day ahead annualized conditional volatility (percentage units)")
    unconditional_annualized: float = Field(..., description="Long-run unconditional annualized volatility (percentage units)")
    trend: Literal["contracting", "expanding", "stable"] = Field(..., description="Trend classification of volatility relative to the long-run baseline")

class RiskMetrics(BaseModel):
    value_at_risk: float = Field(..., description="Percentage Value at Risk (absolute percentage value, e.g. 2.55 for a 2.55% loss)")
    nominal_var: float = Field(..., description="Absolute nominal loss Value at Risk in portfolio currency units")
    expected_shortfall: float = Field(..., description="Percentage Expected Shortfall (absolute percentage value)")
    nominal_expected_shortfall: float = Field(..., description="Absolute nominal loss Expected Shortfall in portfolio currency units")

class ForecastSummary(BaseModel):
    volatility: VolatilitySummary = Field(..., description="Summary of current and historical volatility projections")
    risk: Dict[str, RiskMetrics] = Field(..., description="Risk metrics keyed by confidence level string (e.g. '0.95', '0.99')")

class HorizonForecast(BaseModel):
    horizon_days: int = Field(..., description="Holding period in trading days")
    target_date: date = Field(..., description="The future calendar date corresponding to the end of the horizon")
    cumulative_volatility: float = Field(..., description="Cumulative projected volatility over the horizon")
    annualized_volatility: float = Field(..., description="Annualized projected volatility over the horizon")
    risk_metrics: Dict[str, RiskMetrics] = Field(..., description="Horizon-specific risk metrics keyed by confidence level string")

class ForecastResponse(BaseModel):
    status: Literal["success", "failed"] = Field(default="success", description="Status of the forecast request")
    portfolio_value: float = Field(..., description="Nominal size of the position used in risk estimations")
    summary: ForecastSummary = Field(..., description="Next-day volatility and tail risk diagnostics")
    horizon_forecasts: Optional[List[HorizonForecast]] = Field(
        default=None,
        description="Projections and risk metrics mapped across holding periods"
    )
    model_spec: ModelSpec = Field(..., description="Specification details of the model used for the forecast")
    