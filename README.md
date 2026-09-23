# Financial Time-Series Volatility and Value-at-Risk Forecasting API

This repository delivers a production-grade, mathematically rigorous framework for predicting equity volatility and estimating portfolio tail-risk (Value-at-Risk and Expected Shortfall). 

The application retrieves historical daily stock prices from the **Twelve Data API**, synchronizes records with a local **SQLite database cache**, fits an autoregressive conditional heteroskedasticity (**GARCH**) model via the `arch` library, and exposes high-performance statistical diagnostics and forecasting endpoints via **FastAPI**.

---

## 🚀 Key Features

### 1. Advanced Econometric Modeling & Volatility Forecasting
* **Dynamic Conditional Variance:** Fits a GARCH(p, q) model (with `Zero` mean configuration) to capture volatility clustering and leverage effects in historical log returns.
* **Non-Constant Variance Term-Structure:** Projects multi-period conditional variance by dynamically iterating and aggregating expected daily variances ($\sum_{k=1}^h \sigma^2_{t+k}$), capturing the natural mean-reversion of the GARCH process over longer holding periods.

### 2. Analytical Tail-Risk Estimation (VaR & Expected Shortfall)
* **Dual Distribution Assumptions:** Estimates tail risk under both Gaussian and Student's t-distributions.
* **Student's t-Distribution Unit-Variance Normalization:** Corrects Student-t quantiles using the standard GARCH adjustment factor ($\sqrt{(\nu - 2)/\nu}$) to account for the unit-variance standardization used by solvers, eliminating a common cause of risk underestimation.
* **Analytical Expected Shortfall (ES):** Implements closed-form equations for ES under both Normal and Student's t-distributions to quantify the expected loss in the worst $\alpha\%$ of outcomes.

### 3. Integrated Diagnostic Toolbox
* **Stationarity Testing:** Employs the Augmented Dickey-Fuller (**ADF**) unit root test via `arch.unitroot` to verify stationarity of log returns.
* **Heteroskedasticity Testing:** Implements Engle’s Lagrange Multiplier (**LM**) test via `statsmodels` to confirm the presence of ARCH effects before fitting models.

### 4. Enterprise-Grade Hybrid Data Ingestion & Caching
* **Timezone-Aware Scheduling:** Localizes all times to the New York exchange clock (`America/New_York`) and automatically rolls back requests if the current market is open but today's EOD data is not yet finalized (typically 5:00 PM Eastern).
* **Double-Ended Caching Checks:** Validates local database records on both ends of the lookback window using a 5-day grace window for holidays/weekends. This avoids redundant, slow API calls while guaranteeing that users never train models on stale or incomplete data.

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.8+** or **Docker Desktop**
* **Twelve Data API Key:** Get a free API key from [twelvedata.com](https://twelvedata.com/).

### 1. Configuration
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/yourusername/predicting-stock-volatility-using-Python.git
   cd predicting-stock-volatility-using-Python
   ```
2. Set up your environment file:
   Copy `.sample.env` to `.env` and fill in your API key:
   ```bash
   cp .sample.env .env
   ```
   Modify `.env`:
   ```env
   TWELVE_DATA_API_KEY=your_actual_api_key_here
   ```

### 2. Running with Docker Compose (Recommended)
Docker Compose spins up the FastAPI web service and a Jupyter Notebook environment concurrently:
```bash
docker compose up
```
* **API Swagger Docs:** Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)
* **Jupyter Notebook Demo:** Navigate to [http://localhost:8888](http://localhost:8888) and explore `project-demo.ipynb` using the token shown in your terminal.

### 3. Running with Docker Build
If you only want to build and run the FastAPI server:
```bash
docker build -t garch-api .
docker run -p 8000:8000 garch-api
```

---

## 📡 API Usage Guide

### 1. `GET /hello`
A simple healthcheck endpoint.
* **Response:**
  ```json
  {"message": "Hello, World"}
  ```

---

### 2. `GET /diagnostics/check`
Tests historical returns for stationarity and conditional heteroskedasticity (ARCH effects) to verify if the asset is mathematically suitable for GARCH modeling.
* **Query Parameters:**
  * `identifier` (string, required): Ticker symbol (e.g. `AAPL`) or ISIN.
  * `type` (string, default: `ticker`): `ticker` or `isin`.
  * `period` (string, default: `3y`): Lookback period (`1y`, `3y`, or `5y`).
* **Example Request:**
  ```http
  GET /diagnostics/check?identifier=AAPL&period=3y
  ```
* **Sample Response:**
  ```json
  {
    "id": "AAPL",
    "type": "ticker",
    "date": {
      "start": "2023-09-23",
      "end": "2026-09-23"
    },
    "data": {
      "is_stationary": true,
      "has_arch_effects": true,
      "adf_pvalue": 0.000012,
      "arch_lm_pvalue": 0.000451,
      "recommendation": "Proceed with GARCH(1,1)"
    }
  }
  ```

---

### 3. `POST /models/fit`
Fits a GARCH(p,q) model on historical daily log returns and serializes the trained model artifact to disk.
* **Payload Structure:**
  * `identifier` (string): Asset symbol (e.g., `AAPL`).
  * `window_period` (string, default: `3y`): Historical window (`1y`, `3y`, or `5y`).
  * `parameters` (object): Configures `p` and `q` lags (e.g., `{"p": 1, "q": 1}`).
  * `distribution` (string, default: `normal`): Loss distribution (`normal` or `studentst`).
  * `use_new_data` (boolean, default: `false`): Force bypass the SQLite cache and pull fresh data from the API.
  * `include_coefficients` (boolean, default: `true`): Return parameter coefficients with standard errors and t-stats in the response.
* **Example Request:**
  ```json
  {
    "identifier": "AAPL",
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
    "name": "2026-09-23T17-30-00.123456-GARCH11_AAPL",
    "summary": {
      "model": "GARCH",
      "parameters": { "p": 1, "q": 1 },
      "distribution": "studentst",
      "trained_at": "2026-09-23T17:30:00.123456"
    },
    "data_summary": {
      "start_date": "2023-09-25",
      "end_date": "2026-09-22",
      "total_observations": 754
    },
    "metrics": {
      "log_likelihood": 2245.81,
      "aic": -4481.62,
      "bic": -4458.5,
      "converged": true
    },
    "coefficients": {
      "mu": { "value": 0.00084, "std_err": 0.00038, "t_stat": 2.21, "p_value": 0.027 },
      "omega": { "value": 0.000012, "std_err": 0.000004, "t_stat": 3.0, "p_value": 0.0027 },
      "alpha[1]": { "value": 0.085, "std_err": 0.019, "t_stat": 4.47, "p_value": 0.00001 },
      "beta[1]": { "value": 0.865, "std_err": 0.028, "t_stat": 30.89, "p_value": 0.0 }
    }
  }
  ```

---

### 4. `POST /models/forecast`
Generates next-day and multi-horizon volatility predictions, alongside parametric Value-at-Risk and Expected Shortfall forecasts.
* **Payload Structure:**
  * `identifier` (string): Asset symbol (e.g., `AAPL`).
  * `use_model` (string, default: `latest`): A specific model filename or `latest` to load the most recent fitted model.
  * `portfolio_value` (float, default: `10000.0`): Portfolio position size in nominal terms.
  * `annualization_factor` (integer, default: `252`): Base factor (e.g., `252` for equities, `365` for crypto).
  * `horizon` (list of integers, optional): Holding periods in trading days (e.g., `[1, 5, 20, 60]`).
  * `value_at_risk` (object): Configures confidence thresholds (e.g., `{"confidence_levels": [0.95, 0.99]}`).
* **Example Request:**
  ```json
  {
    "identifier": "AAPL",
    "use_model": "latest",
    "portfolio_value": 1000000.0,
    "annualization_factor": 252,
    "horizon": [5, 20],
    "value_at_risk": {
      "confidence_levels": [0.95, 0.99]
    }
  }
  ```
* **Sample Response Body:**
  ```json
  {
    "status": "success",
    "portfolio_value": 1000000.0,
    "summary": {
      "volatility": {
        "conditional_next_day": 1.45,
        "conditional_annualized": 22.98,
        "unconditional_annualized": 24.12,
        "trend": "expanding"
      },
      "risk": {
        "0.95": {
          "value_at_risk": -0.0238,
          "nominal_var": 23800.0,
          "expected_shortfall": -0.0298,
          "nominal_expected_shortfall": 29800.0
        },
        "0.99": {
          "value_at_risk": -0.0336,
          "nominal_var": 33600.0,
          "expected_shortfall": -0.0385,
          "nominal_expected_shortfall": 38500.0
        }
      }
    },
    "horizon_forecasts": [
      {
        "horizon_days": 5,
        "target_date": "2026-09-30",
        "cumulative_volatility": 3.24,
        "annualized_volatility": 22.99,
        "risk_metrics": {
          "0.95": {
            "value_at_risk": -0.0533,
            "nominal_var": 53300.0,
            "expected_shortfall": -0.0667,
            "nominal_expected_shortfall": 66700.0
          }
        }
      }
    ],
    "model_spec": {
      "model_name": "2026-09-23T17-30-00.123456-GARCH11_AAPL",
      "model_type": "GARCH",
      "order": { "p": 1, "q": 1 },
      "distribution": "studentst",
      "trained_at": "2026-09-23T17:30:00.123456",
      "data": {
        "identifier": "AAPL",
        "last_price_date": "2026-09-22"
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
