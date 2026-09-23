import math
from typing import Dict, Tuple, List
import pandas as pd
from scipy import stats
import numpy as np
from datetime import date, timedelta
import schemas as scm

def get_volatility_summary(res, annualization_factor) -> scm.VolatilitySummary:
    """
    Computes 1-day conditional volatility, unconditional long-run volatility,
    and classifies trend as expanding or contracting.
    
    Note: Data items in % units
    """
    params = res.params
    omega = params.get("omega", 0.0) 
    
    # Sum of ARCH (alpha) and GARCH (beta) persistence terms
    alpha_sum = sum(val for key, val in params.items() if key.startswith("alpha"))
    beta_sum = sum(val for key, val in params.items() if key.startswith("beta"))
    persistence = alpha_sum + beta_sum

    if persistence < 0.9999 and persistence > 0:
        long_run_var_daily = omega / (1.0 - persistence)
        uncond_vol_ann = math.sqrt(long_run_var_daily) * math.sqrt(annualization_factor)
    else:
        # Fallback to sample variance of residuals if non-stationary
        uncond_vol_ann = float(np.std(res.resid)) * math.sqrt(annualization_factor)

    # 1-day ahead conditional forecast
    forecast_res = res.forecast(horizon=1, reindex=False)
    next_day_daily = math.sqrt(forecast_res.variance.values[-1, 0]) 
    next_day_ann = next_day_daily * math.sqrt(annualization_factor)

    # Trend relative to equilibrium
    if next_day_ann > (uncond_vol_ann * 1.02):
        trend = "contracting"  # Elevated above baseline; expected to mean-revert down
    elif next_day_ann < (uncond_vol_ann * 0.98):
        trend = "expanding"    # Below baseline; expected to expand up
    else:
        trend = "stable"

    # volatility_summary =  {
    #     'conditional_next_day': round(next_day_daily, 2),
    #     'conditional_annualized': round(next_day_ann, 2),
    #     'unconditional_annualized': round(uncond_vol_ann, 2),
    #     'trend': trend
    # }  
    
    volatility_summary = scm.VolatilitySummary(
        conditional_next_day=round(next_day_daily, 2),
        conditional_annualized=round(next_day_ann, 2),
        unconditional_annualized=round(uncond_vol_ann, 2),
        trend=trend
    )
    
    return volatility_summary
    
def compute_distribution_quantiles(
    dist_name: str,
    params: pd.Series,
    alpha: float,
) -> Tuple[float, float]:
    """
    Computes standard cutoff quantile (q) and Expected Shortfall factor (es_factor)
    such that:
      VaR = q * sigma
      ES  = es_factor * sigma
    under zero-mean assumption. Both return negative values for tail losses.
    """
    dist_clean = dist_name.lower()

    if dist_clean in ["studentst", "t"]:
        # Degrees of freedom estimated by arch is typically named 'nu'
        df = float(params.get("nu", 8.0))
        if df <= 2.0:
            df = 2.01  # Variance undefined for nu <= 2

        # arch scales t residuals to unit variance: z = r / (sigma * sqrt((df-2)/df))
        scale_adj = math.sqrt((df - 2.0) / df)
        
        # Standard Student-t quantile
        t_q = stats.t.ppf(1.0 - alpha, df)
        q = float(t_q * scale_adj)

        # Expected Shortfall for Student's t: E[Z | Z < -|q|]
        # Formula: - ( (df + t_q^2) / (df - 1) ) * ( f(t_q) / (1 - alpha) ) * scale_adj
        pdf_val = stats.t.pdf(t_q, df)
        es_raw = -((df + t_q**2) / (df - 1.0)) * (pdf_val / alpha)
        es_factor = float(es_raw * scale_adj)

    else:
        # Standard Gaussian default
        q = float(stats.norm.ppf(1.0 - alpha))
        pdf_val = stats.norm.pdf(q)
        es_factor = float(-pdf_val / alpha)

    return -abs(q), -abs(es_factor)

def calculate_risk_metrics_for_horizon(
    cumulative_vol: float,
    confidence_levels: list[float],
    portfolio_value: float,
    dist_name: str,
    params: pd.Series,
) -> Dict[str, scm.RiskMetrics]:
    """Generates RiskMetricValues dictionary keyed by confidence level string."""
    metrics = {}

    for conf in confidence_levels:
        alpha = 1 - conf
        q, es_factor = compute_distribution_quantiles(dist_name, params, alpha)
        
        var_pct = abs(q * cumulative_vol)
        es_pct = abs(es_factor * cumulative_vol)

        conf_key = f"{conf:.2f}"
        # metrics[conf_key] = {
        #     'value_at_risk': round(var_pct, 6),
        #     'nominal_var': round((var_pct * portfolio_value)/100, 2),
        #     'expected_shortfall': round(es_pct, 6),
        #     'nominal_expected_shortfall': round((es_pct * portfolio_value)/100, 2)
        # }
        
        metrics[conf_key] = scm.RiskMetrics(
            value_at_risk=round(var_pct, 4),
            nominal_var=round((var_pct * portfolio_value)/100, 2),
            expected_shortfall=round(es_pct, 4),
            nominal_expected_shortfall=round((es_pct * portfolio_value)/100, 2)
        )
    return metrics

def advance_business_days(start_date: date, days_to_add: int) -> date:
    """Advances a date by N business days, skipping Saturday and Sunday."""
    current = start_date
    added = 0
    while added < days_to_add:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current


def get_horizon_forecasts(payload, 
                          res, 
                          daily_variances, 
                          distribution, 
                          last_price_date) -> List[scm.HorizonForecast]:
    horizon_forecasts = []
    horizon = payload.horizon
    annualization_factor= payload.annualization_factor
    confidence_levels = payload.value_at_risk.confidence_levels
    portfolio_value = payload.portfolio_value
    
    for h in horizon:
        # Cumulative variance over h days = sum of daily conditional variances
        cum_variance = float(np.sum(daily_variances[:h]))
        cum_vol = math.sqrt(cum_variance)

        # Annualized equivalent over the h-day span
        ann_vol = (cum_vol / math.sqrt(h)) * math.sqrt(annualization_factor)
        target_date = advance_business_days(last_price_date, h)
        
        print(f'Day {h} annual volatility: {ann_vol}')
        print(f'Day {h} target date: {target_date}')

        h_risk_metrics = calculate_risk_metrics_for_horizon(
            cumulative_vol=cum_vol,
            confidence_levels=confidence_levels,
            portfolio_value=portfolio_value,
            dist_name=distribution,
            params=res.params,
        )

        # h_forecast = {
        #     'horizon_days': h,
        #     'target_date': target_date,
        #     'cumulative_volatility': round(cum_vol, 6),
        #     'annualized_volatility': round(ann_vol, 6),
        #     'risk_metrics': h_risk_metrics
        # }
        
        h_forecast = scm.HorizonForecast(
            horizon_days=h,
            target_date=target_date,
            cumulative_volatility=round(cum_vol, 4),
            annualized_volatility=round(ann_vol, 4),
            risk_metrics=h_risk_metrics
        )
        
        horizon_forecasts.append(h_forecast)

    return horizon_forecasts