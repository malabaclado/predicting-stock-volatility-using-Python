# Financial Time-Series Volatility and Value-at-Risk Forecasting API

This repository delivers a production-grade, mathematically rigorous framework for predicting equity volatility and estimating portfolio tail-risk (Value-at-Risk and Expected Shortfall). 

The application retrieves historical daily stock prices from the **Twelve Data API**, synchronizes records with a local **SQLite database cache (`market_data.sqlite`)**, fits an autoregressive conditional heteroskedasticity (**GARCH**) model via the `arch` library, records model metadata and performance metrics in a registry database (**`models.sqlite`**), and exposes high-performance statistical diagnostics and forecasting endpoints via **FastAPI**.

---

## 🚀 Key Features

### 1.  Econometric Modeling & Volatility Forecasting
* **Dynamic Conditional Variance:** Fits a GARCH(p, q) model (with `Zero` mean configuration) to capture volatility clustering and leverage effects in historical log returns.
* **Non-Constant Variance Term-Structure:** Projects multi-period conditional variance by dynamically iterating and aggregating expected daily variances ($\sum_{k=1}^h \sigma^2_{t+k}$), capturing the natural mean-reversion of the GARCH process over longer holding periods.

### 2. Analytical Tail-Risk Estimation (VaR & Expected Shortfall)
* **Dual Distribution Assumptions:** Estimates tail risk under both Gaussian and Student's t-distributions.
* **Student's t-Distribution Unit-Variance Normalization:** Corrects Student-t quantiles using the standard GARCH adjustment factor ($\sqrt{(\nu - 2)/\nu}$) to account for the unit-variance standardization used by solvers, eliminating a common cause of risk underestimation.
* **Analytical Expected Shortfall (ES):** Implements closed-form equations for ES under both Normal and Student's t-distributions to quantify the expected loss in the worst $\alpha\%$ of outcomes.

### 3. Integrated Diagnostic Toolbox
* **Stationarity Testing:** Employs the Augmented Dickey-Fuller (**ADF**) unit root test via `arch.unitroot` to verify stationarity of log returns.
* **Heteroskedasticity Testing:** Implements Engle’s Lagrange Multiplier (**LM**) test via `statsmodels` to confirm the presence of ARCH effects before fitting models.

### 4. Persistent Model Registry & Search
* **Cataloged Model Metadata:** Automatically logs every fitted model into the SQLite `models` table (`models.sqlite`) with AIC, BIC, persistence ($\alpha + \beta$), convergence status, and training date.
* **Multi-Criteria Search Endpoint:** Query saved models by ticker, convergence, date ranges, persistence thresholds, or distribution, with sorting by AIC, BIC, persistence, or training timestamp.

### 5. Enterprise-Grade Hybrid Data Ingestion & Caching
* **Timezone-Aware Scheduling:** Localizes all times to the New York exchange clock (`America/New_York`) and automatically rolls back requests if the current market is open but today's EOD data is not yet finalized (typically 5:00 PM Eastern).
* **Double-Ended Caching Checks:** Validates local database records on both ends of the lookback window using a grace window for holidays/weekends. This avoids redundant, slow API calls while guaranteeing that users never train models on stale or incomplete data.

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.10+** or **Docker Desktop**
* **Twelve Data API Key:** Get a free API key from [twelvedata.com](https://twelvedata.com/).

### 1. Configuration
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/yourusername/predicting-stock-volatility-using-Python.git
   cd predicting-stock-volatility-using-Python
   ```
2. Set up your environment file:
   Copy `.sample.env` to `.env` and fill in your API key and configuration:
   ```bash
   cp .sample.env .env
   ```
   Modify `.env`:
   ```env
   TWELVE_DATA_API_KEY=your_actual_api_key_here
   ```

### 2. Running Locally with Python
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger documentation.

### 3. Running with Docker Compose (Recommended)
Docker Compose spins up the FastAPI web service and a Jupyter Notebook environment concurrently:
```bash
docker compose up
```
* **API Swagger Docs:** Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)
* **Jupyter Notebook Demo:** Navigate to [http://localhost:8888](http://localhost:8888) and explore `project-demo.ipynb`.

### 4. Running with Docker Build
If you only want to build and run the FastAPI server:
```bash
docker build -t garch-api .
docker run -p 8000:8000 garch-api
```

---

## 📡 API Usage Guide

### 1. `GET /diagnostics/check`
Tests historical returns for stationarity (ADF test) and conditional heteroskedasticity (Engle's ARCH LM test) to verify if the equity time series is mathematically suitable for GARCH modeling.

* **Query Parameters:**
  * `ticker` (string, required): Equity ticker symbol (e.g. `AAPL`).
  * `period` (string, default: `3y`): Lookback period (`1y`, `3y`, or `5y`).
* **Example Request:**
  ```http
  GET /diagnostics/check?ticker=AAPL&period=3y
  ```
* **Sample Response:**
  ```json
  {
    "ticker": "AAPL",
    "date": {
      "start": "2023-10-02",
      "end": "2026-09-29"
    },
    "data": {
      "is_stationary": true,
      "has_arch_effects": true,
      "adf_pvalue": 8.88e-26,
      "arch_lm_pvalue": 5.14e-21,
      "recommendation": "Proceed with GARCH(1,1)"
    }
  }
  ```

---

### 2. `POST /model/search`
Searches and filters saved model artifacts cataloged in the SQLite registry (`models.sqlite`) based on performance metrics, training dates, and convergence.

* **Payload Structure:**
  * `ticker` (string, optional): Filter by equity ticker (e.g. `"AAPL"`).
  * `converged` (boolean, default: `true`): Filter by optimizer convergence status.
  * `trained_at` (object, optional): Date range filter (`{"start": "2026-01-01", "end": "2026-09-30"}`).
  * `persistence` (object, optional): Volatility persistence range (`{"min": 0.0, "max": 0.999}`).
  * `distribution` (string, default: `studentst`): Error distribution filter (`normal` or `studentst`).
  * `window_period` (string, optional): Lookback period (`1y`, `3y`, or `5y`).
  * `limit` (integer, required): Maximum number of records to return (1 to 200).
  * `sort_by` (string, required): Sorting attribute (`"aic"`, `"bic"`, `"trained_at"`, `"persistence"`).
  * `order_by` (string, optional): `"asc"` or `"desc"` (defaults to `"asc"` for aic/bic and `"desc"` for trained_at/persistence).
* **Example Request:**
  ```json
  {
    "ticker": "AAPL",
    "converged": true,
    "distribution": "studentst",
    "limit": 5,
    "sort_by": "aic"
  }
  ```
* **Sample Response Body:**
  ```json
  {
    "search_results": [
      {
        "model_name": "2026-09-30T10-47-01.396909-GARCH11_AAPL",
        "model_type": "GARCH",
        "order": { "p": 1, "q": 1 },
        "distribution": "studentst",
        "trained_at": "2026-09-30",
        "data": {
          "ticker": "AAPL",
          "start_date": "2023-10-02",
          "end_date": "2026-09-29",
          "total_observations": 750,
          "window_period": "3y"
        },
        "metrics": {
          "aic": 2705.6937,
          "bic": 2724.174,
          "persistence": 0.7959,
          "converged": true
        }
      }
    ],
    "total": 1
  }
  ```

---

### 3. `POST /models/fit`
Fits a GARCH(p,q) volatility model on historical daily log returns, dumps the artifact (`.pkl`) to disk, and records its metadata into `models.sqlite`.

* **Payload Structure:**
  * `ticker` (string, required): Asset ticker symbol (e.g., `AAPL`).
  * `window_period` (string, default: `3y`): Historical window (`1y`, `3y`, or `5y`).
  * `parameters` (object): Configures `p` and `q` lags (e.g., `{"p": 1, "q": 1}`).
  * `distribution` (string, default: `studentst`): Error distribution (`normal` or `studentst`).
  * `use_new_data` (boolean, default: `false`): Force bypass the SQLite cache and pull fresh data from the API.
  * `include_coefficients` (boolean, default: `false`): Return parameter estimates with standard errors and t-stats.
* **Example Request:**
  ```json
  {
    "ticker": "AAPL",
    "window_period": "3y",
    "parameters": { "p": 1, "q": 1 },
    "distribution": "studentst",
    "use_new_data": false,
    "include_coefficients": true
  }
  ```
* **Sample Response Body:**
  ```json
  {
    "status": "success",
    "name": "2026-09-30T10-47-01.396909-GARCH11_AAPL",
    "summary": {
      "model": "GARCH",
      "parameters": { "p": 1, "q": 1 },
      "distribution": "studentst",
      "trained_at": "2026-09-30 10:47:01.396909"
    },
    "data": {
      "ticker": "AAPL",
      "start_date": "2023-10-02",
      "end_date": "2026-09-29",
      "total_observations": 750,
      "window_period": "3y"
    },
    "metrics": {
      "log_likelihood": -1348.8469,
      "aic": 2705.6937,
      "bic": 2724.174,
      "converged": true
    },
    "coefficients": {
      "omega": { "value": 0.3642, "std_err": 0.1741, "t_stat": 2.0915, "p_value": 0.0365 },
      "alpha[1]": { "value": 0.1333, "std_err": 0.0949, "t_stat": 1.4042, "p_value": 0.1603 },
      "beta[1]": { "value": 0.6626, "std_err": 0.294, "t_stat": 2.2539, "p_value": 0.0242 },
      "nu": { "value": 3.4885, "std_err": 0.4573, "t_stat": 7.6282, "p_value": 0.0 }
    }
  }
  ```

---

### 4. `POST /models/forecast`
Generates next-day and multi-horizon volatility predictions, alongside parametric Value-at-Risk and Expected Shortfall forecasts.

* **Payload Structure:**
  * `ticker` (string, required): Asset ticker symbol (e.g., `AAPL`).
  * `use_model` (string, default: `latest`): Specific model artifact name or `latest` to use the most recent fitted model.
  * `portfolio_value` (float, default: `10000.0`): Portfolio position size in nominal account currency.
  * `annualization_factor` (integer, default: `252`): Base trading days multiplier (e.g., `252` for equities).
  * `horizon` (list of integers, optional): Holding periods in trading days (e.g., `[1, 5, 20]`).
  * `value_at_risk` (object): Configures confidence thresholds (e.g., `{"confidence_levels": [0.95, 0.99]}`).
* **Example Request:**
  ```json
  {
    "ticker": "AAPL",
    "use_model": "latest",
    "portfolio_value": 50000.0,
    "annualization_factor": 252,
    "horizon": [1, 5, 20],
    "value_at_risk": {
      "confidence_levels": [0.95, 0.99]
    }
  }
  ```
* **Sample Response Body:**
  ```json
  {
    "status": "success",
    "portfolio_value": 50000.0,
    "summary": {
      "volatility": {
        "conditional_next_day": 1.5413,
        "conditional_annualized": 24.4661,
        "unconditional_annualized": 27.6083,
        "trend": "contracting"
      },
      "risk": {
        "0.95": {
          "value_at_risk": 2.7153,
          "nominal_var": 1357.65,
          "expected_shortfall": 4.3812,
          "nominal_expected_shortfall": 2190.6
        },
        "0.99": {
          "value_at_risk": 4.8964,
          "nominal_var": 2448.2,
          "expected_shortfall": 7.2144,
          "nominal_expected_shortfall": 3607.2
        }
      }
    },
    "horizon_forecasts": [
      {
        "horizon_days": 1,
        "target_date": "2026-09-30",
        "cumulative_volatility": 1.5413,
        "annualized_volatility": 24.4661,
        "risk_metrics": {
          "0.95": {
            "value_at_risk": 2.7153,
            "nominal_var": 1357.65,
            "expected_shortfall": 4.3812,
            "nominal_expected_shortfall": 2190.6
          }
        }
      },
      {
        "horizon_days": 5,
        "target_date": "2026-10-06",
        "cumulative_volatility": 3.4735,
        "annualized_volatility": 24.663,
        "risk_metrics": {
          "0.95": {
            "value_at_risk": 6.1187,
            "nominal_var": 3059.35,
            "expected_shortfall": 9.8732,
            "nominal_expected_shortfall": 4936.6
          }
        }
      }
    ],
    "model_spec": {
      "model_name": "2026-09-30T10-47-01.396909-GARCH11_AAPL",
      "model_type": "GARCH",
      "order": { "p": 1, "q": 1 },
      "distribution": "studentst",
      "trained_at": "2026-09-30 10:47:01.396909",
      "data": {
        "ticker": "AAPL",
        "start_date": "2023-10-02",
        "end_date": "2026-09-29",
        "total_observations": 750,
        "window_period": "3y"
      }
    }
  }
  ```

---

## 📈 Mathematics and Modeling Reference

### 1. Daily Log Returns
Daily prices are transformed into stationary log returns:
$$R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### 2. GARCH(1,1) Conditional Variance
$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$
* **$\omega$ (omega):** Constant baseline variance.
* **$\alpha_1$ (alpha):** Sensitivity to recent market shocks (ARCH term).
* **$\beta_1$ (beta):** Persistence of historical volatility (GARCH term).
* **Covariance Stationarity Constraint:** $\alpha_1 + \beta_1 < 1$.

### 3. Expected Shortfall (ES) Analytical Formula
Under the zero-mean assumption, the conditional ES for standard Normal residuals ($Z \sim N(0,1)$) at significance level $\alpha = 1 - c$ is calculated as:
$$\text{ES}_\alpha(Z) = -\frac{\phi(z_\alpha)}{\alpha}$$
where $\phi$ is the standard Normal PDF and $z_\alpha$ is the normal quantile. 

Under Student's t-distribution standardized to unit-variance, the conditional ES is:
$$\text{ES}_\alpha(Z) = -\left(\frac{\nu + t_\alpha^2}{\nu - 1}\right) \frac{f(t_\alpha)}{\alpha} \times \sqrt{\frac{\nu - 2}{\nu}}$$
where $f$ is the Student's t PDF, $t_\alpha$ is the Student's t quantile, and $\nu$ is the degrees of freedom.

> **Note on Sign Convention:** Consistent with institutional risk management practice, the API presents VaR and Expected Shortfall as positive loss quantities ($\text{VaR}_{\%} = |q| \times \sigma_{\text{cum}}$ and $\text{ES}_{\%} = |\text{ES}_{\text{factor}}| \times \sigma_{\text{cum}}$) alongside nominal currency loss figures.
