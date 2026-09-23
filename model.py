import os
import sqlite3
from glob import glob

import pickle
import pandas as pd
import numpy as np
from arch import arch_model
from config import settings

from data import TwelveDataAPI, SQLRepository, get_start_date, get_latest_expected_eod


def build_model(ticker: str) -> object:
    """
    Initializes database connection, creates a SQLRepository object and a GarchModel object.
    """
    # Create DB connection
    connection = sqlite3.connect(database=settings.db_name, check_same_thread=False)

    # Create `SQLRepository`
    repo = SQLRepository(connection)

    # Create model
    model = GarchModel(ticker, repo)

    # Return model
    return model


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
            
            # Ensure idType gets the proper string (e.g. self.id_type or 'ticker'), not the built-in `type`
            id_type = getattr(self, "id_type", "ticker")
            df = api.fetch_data_from_api(
                identifier, start_date_str, idType=id_type
            )

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
        
        if model_name == "latest":
            if not os.path.exists(MODEL_STORE_DIR):
                raise FileNotFoundError("Model storage directory does not exist.")
            files = [f for f in os.listdir(MODEL_STORE_DIR) if f.endswith(".pkl")]
            if not files:
                raise FileNotFoundError("No fitted model artifacts found.")
            model_name = sorted(files)[-1].replace(".pkl", "")

        file_path = os.path.join(MODEL_STORE_DIR, f"{model_name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model artifact '{model_name}' not found.")

        with open(file_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact['model_result']
        self.name = model_name
        print(f"Loading {model_name} success!")
        
        return artifact

        
    