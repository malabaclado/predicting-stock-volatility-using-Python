import os
import sqlite3
from glob import glob

import pickle
import pandas as pd
import numpy as np
from arch import arch_model
from config import settings

from src.data import TwelveDataAPI, SQLRepository, get_start_date, get_latest_expected_eod

def build_model(ticker: str) -> object:
    """
    Initializes database connection, creates a SQLRepository object and a GarchModel object.
    """
    # Create DB connection
    connection = sqlite3.connect(database=settings.data_db_name, check_same_thread=False)

    # Create `SQLRepository`
    repo = SQLRepository(connection)

    # Create model
    model = GarchModel(ticker, repo)

    # Return model
    return model

def read_models_table() -> pd.DataFrame:
    """Reads and returns the models table from the sqlite database."""
    with sqlite3.connect(database=settings.models_db_name, check_same_thread=False) as conn:
        df = pd.read_sql("SELECT * FROM models", conn, index_col="model_name")
        
    # Set column types
    df = df.astype({
        "ticker": str,
        "distribution": str,
        "window_period": str
    })

    df["trained_at"] = pd.to_datetime(df["trained_at"]).dt.date
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
        
    return df

def filter_saved_models(df, payload):
    ticker = payload.ticker
    converged = payload.converged
    trained_at_start = payload.trained_at.start
    trained_at_end = payload.trained_at.end
    min_persistence = payload.persistence.min
    max_persistence = payload.persistence.max
    distribution = payload.distribution
    window_period = payload.window_period
    limit = payload.limit
    sort_by = payload.sort_by
    order_by = payload.order_by
    
    # 1. Start with an all-True mask matching df's index
    mask = pd.Series(True, index=df.index)
    
    if ticker is not None:
        mask &= df["ticker"] == ticker
    if distribution is not None:
            mask &= df["distribution"] == distribution
    if window_period is not None:
        mask &= df["window_period"] == window_period
        
        
    if converged:
        mask &= df["converged"] == 0
    else:
        mask &= df["converged"] != 0
        
    if trained_at_start is not None:
        mask &= df["trained_at"] >= trained_at_start
    if trained_at_end is not None:
        mask &= df["trained_at"] <= trained_at_end
            
    if min_persistence is not None:
        mask &= df["persistence"] >= min_persistence
    if max_persistence is not None:
            mask &= df["persistence"] <= max_persistence
        
    filtered_df = df[mask]
    
    total = len(filtered_df)
    is_ascending = (order_by == 'asc')
    
    return total, filtered_df.sort_values(by=sort_by, ascending=is_ascending).head(limit)


def save_model_to_db(record: dict) -> dict:
    """Inserts a single model record into the SQLite models table using SQLRepository.

    Parameters
    ----------
    record : dict
        Dictionary containing model metadata with 'model_name' as key or field.

    Returns
    -------
    dict
        The inserted record dictionary.
    """
    df = pd.DataFrame([record])
    if "model_name" in df.columns:
        df.set_index("model_name", inplace=True)

    with sqlite3.connect(database=settings.models_db_name, check_same_thread=False) as conn:
        # Prevent duplicate rows if model_name already exists
        model_name = record.get("model_name")
        if model_name:
            cursor = conn.cursor()
            table_check = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
            ).fetchone()
            if table_check:
                cursor.execute("DELETE FROM models WHERE model_name = ?", (model_name,))
                conn.commit()

        repo = SQLRepository(conn)
        repo.insert_table(table_name="models", records=df, if_exists="append")

    return record

class GarchModel():
    """Class for training GARCH model and generating predictions.

    Atttributes
    -----------
    ticker : str
        Ticker symbol of the equity whose volatility will be predicted.
    repo : SQLRepository
        The repository where the training data will be stored.
    use_new_data : bool
        Whether to download new data from the AlphaVantage API to train
        the model or to use the existing data stored in the repository.
    model_directory : str
        Path for directory where trained models will be stored.

    Methods
    -------
    wrangle_data
        Generate equity returns from data in database.
    fit
        Fit model to training data.
    predict
        Generate volatilty forecast from trained model.
    dump
        Save trained model to file.
    load
        Load trained model from file.
    """

    def __init__(self, ticker, repo):
        self.ticker = ticker
        self.repo = repo
        self.model_directory = settings.model_directory
        
    def _check_cache(self, start_date_str: str):
        '''
        Checks cache and returns cached data if valid, otherwise None
        '''
        identifier = self.ticker
        connection = self.repo
        
        earliest_expected_date = pd.to_datetime(start_date_str)
        latest_expected_date = get_latest_expected_eod()
        end_date_str = latest_expected_date.strftime("%Y-%m-%d")
        
        # Set df to None by default
        df = None
        
        # 1. Try reading from SQLite cache
        try:
            cached_df = connection.read_table(
                identifier, start_date_str, end_date_str
            )
            if not cached_df.empty:
                # Normalize index to DatetimeIndex for accurate date comparison
                if not isinstance(cached_df.index, pd.DatetimeIndex):
                    cached_df.index = pd.to_datetime(cached_df.index)

                # Check if the cache covers the full requested lookback period
                earliest_cached = cached_df.index.min()
                latest_cached = cached_df.index.max()
                
                is_start_valid = earliest_cached <= earliest_expected_date 
                is_end_valid = latest_cached >= latest_expected_date
                
                if is_start_valid and is_end_valid:
                    df = cached_df
                else:
                    reason = []
                    if not is_start_valid:
                        reason.append("start date incomplete")
                    if not is_end_valid:
                        reason.append("end date incomplete/stale")
                    print(
                        f"[Cache Incomplete ({', '.join(reason)})] "
                        f"Requested range: {earliest_expected_date.date()} to {latest_expected_date.date()}, "
                        f"cached range: {earliest_cached.date()} to {latest_cached.date()}. "
                        f"Fetching fresh data..."
                    )
        except (ValueError, Exception) as exc:
            print(f"[Cache Miss] {exc}. Fetching from API...")
        
        return df
        

    def get_daily_returns(self, period: str = "3y", use_new_data=False) -> pd.Series:
        identifier = self.ticker
        connection = self.repo
        start_date_str = get_start_date(period)
        earliest_expected_date = pd.to_datetime(start_date_str)
        latest_expected_date = get_latest_expected_eod()
        end_date_str = latest_expected_date.strftime("%Y-%m-%d")

        df = None
        
        if not use_new_data:
            # 1. Try reading from SQLite cache
            try:
                cached_df = connection.read_table(
                    identifier, start_date_str, end_date_str
                )
                if not cached_df.empty:
                    # Normalize index to DatetimeIndex for accurate date comparison
                    if not isinstance(cached_df.index, pd.DatetimeIndex):
                        cached_df.index = pd.to_datetime(cached_df.index)

                    # Check if the cache covers the full requested lookback period
                    earliest_cached = cached_df.index.min()
                    latest_cached = cached_df.index.max()
                    
                    is_start_valid = earliest_cached <= earliest_expected_date 
                    is_end_valid = latest_cached >= latest_expected_date
                    
                    if is_start_valid and is_end_valid:
                        df = cached_df
                    else:
                        reason = []
                        if not is_start_valid:
                            reason.append("start date incomplete")
                        if not is_end_valid:
                            reason.append("end date incomplete/stale")
                        print(
                            f"[Cache Incomplete ({', '.join(reason)})] "
                            f"Requested range: {earliest_expected_date.date()} to {latest_expected_date.date()}, "
                            f"cached range: {earliest_cached.date()} to {latest_cached.date()}. "
                            f"Fetching fresh data..."
                        )
            except (ValueError, Exception) as exc:
                print(f"[Cache Miss] {exc}. Fetching from API...")
        

        # 2. Fetch from API if cache was missing or incomplete
        if df is None or df.empty:
            # Initialize TwelveDataAPI object
            api = TwelveDataAPI()
            df = api.fetch_data_from_api(identifier, start_date_str)

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 3. Save or update cache in database
            connection.insert_table(identifier, df)
            print("Data saved to /market_data.sqlite")

        # 4. Filter to exact requested window
        df = df.loc[
            (df.index >= earliest_expected_date)
            & (df.index <= pd.to_datetime(latest_expected_date))
        ].copy()

        # 5. Sort chronologically
        if "date" in df.columns:
            df.sort_values(by="date", ascending=True, inplace=True)
        else:
            df.sort_index(ascending=True, inplace=True)

        # 6. Calculate Log Returns
        # dropna() removes the first NaN caused by shifting
        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

        self.data = log_returns
        return

    def fit(self, p, q, dist):

        """Create model, fit to `self.data`, and attach to `self.model` attribute.
        For assignment, also assigns adds metrics to `self.aic` and `self.bic`.

        Parameters
        ----------
        p : int
            Lag order of the symmetric innovation

        q : ind
            Lag order of lagged volatility

        Returns
        -------
        None
        """

        # Hard-coded GARCH variables
        daily_log_returns = self.data
        scaled_returns = daily_log_returns * 100
        
        model = arch_model(
                scaled_returns, 
                p=p,
                q=q, 
                mean="Zero",
                dist=dist,
                rescale = False
            ).fit(disp=0)
        
        
        # Train Model, attach to `self.model`
        self.p = p
        self.q = q
        self.model = model
        self.trained_date = model._datetime
        
        return model
        
        

    def __clean_prediction(self, prediction):

        """Reformat model prediction to JSON.

        Parameters
        ----------
        prediction : pd.DataFrame
            Variance from a `ARCHModelForecast`

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Calculate forecast start date
        start = prediction.index[0] + pd.DateOffset(days=1)

        # Create date range
        prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])
        

        # Create prediction index labels, ISO 8601 format
        prediction_index = [d.isoformat() for d in prediction_dates]


        # Extract predictions from DataFrame, get square root
        data= (prediction**0.5).iloc[0,:]
    
        # Combine `data` and `prediction_index` into Series
        data.index=prediction_index


        # Return Series as dictionary
        return data.to_dict()

    def predict_volatility(self, horizon=5):

        """Predict volatility using `self.model`

        Parameters
        ----------
        horizon : int
            Horizon of forecast, by default 5.

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Generate variance forecast from `self.model`
        prediction = self.model.forecast(horizon=horizon, reindex=False).variance

        # Format prediction with `self.__clean_predction`
        prediction_formatted = self.__clean_prediction(prediction)

        # Return `prediction_formatted`
        return prediction_formatted


    def dump(self, artifact):

        """Save model to `self.model_directory` with timestamp.

        Returns
        -------
        str
            filepath where model was saved.
        """
        # Ensure the directory exists before saving
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Create timestamp in ISO format
        # timestamp = pd.Timestamp.now().isoformat()
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    
        # Create filepath, including `self.model_directory`
        model_name = f"{timestamp}-GARCH{self.p}{self.q}_{self.ticker}"
        file_path = os.path.join(self.model_directory, f"{model_name}.pkl")
        
        # Add model_name to artifact
        artifact['model_name'] = model_name

        with open(file_path, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Return name and filepath
        return model_name
    

    def load(self, model_name):

        """Load most recent model in `self.model_directory` for `self.ticker`,
        attach to `self.model` attribute.

        """
        MODEL_STORE_DIR = settings.model_directory
        ticker_clean = self.ticker.strip().upper()
        
        if model_name == "latest":
            if not os.path.exists(MODEL_STORE_DIR):
                raise FileNotFoundError("Model storage directory does not exist.")
            expected_suffix = f"_{ticker_clean}.pkl".lower()
            files = [
                f for f in os.listdir(MODEL_STORE_DIR)
                if f.lower().endswith(expected_suffix)
            ]
            if not files:
                raise FileNotFoundError(f"No fitted model artifacts found for ticker '{ticker_clean}'.")
            model_name = sorted(files)[-1].replace(".pkl", "")

        file_path = os.path.join(MODEL_STORE_DIR, f"{model_name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model artifact '{model_name}' not found.")

        with open(file_path, "rb") as f:
            artifact = pickle.load(f)

        artifact_ticker = str(artifact.get('ticker') or artifact.get('identifier') or '').strip().upper()
        if artifact_ticker and artifact_ticker != ticker_clean:
            raise ValueError(
                f"Model artifact '{model_name}' was trained on '{artifact_ticker}', "
                f"not on requested ticker '{ticker_clean}'."
            )

        self.model = artifact['model_result']
        self.name = model_name
        print(f"Loading {model_name} success!")
        
        return artifact

        
    