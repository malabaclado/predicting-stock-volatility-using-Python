import os
from glob import glob

import joblib
import pandas as pd
from arch import arch_model
from config import settings
from data import AlphaVantageAPI, SQLRepository


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

    def __init__(self, ticker, repo, use_new_data):
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory

    def wrangle_data(self, n_observations):

        """Extract data from database (or get from AlphaVantage), transform it
        for training model, and attach it to `self.data`.

        Parameters
        ----------
        n_observations : int
            Number of observations to retrieve from database

        Returns
        -------
        None
        """
        # Add new data to database if required
        if self.use_new_data:
            #Instantiate ALphaVantage API class
            api= AlphaVantageAPI()
            
            # Get the data
            new_data = api.get_daily(ticker=self.ticker)
            
            # Insert the data to self.repo
            self.repo.insert_table(table_name=self.ticker, records=new_data, if_exists='replace')

        # Pull data from SQL database
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations)
        

        # Clean data, attach to class as `data` attribute
        df.sort_values(by="date", ascending=True, inplace=True)
        df["return"] = df["close"].pct_change() * 100

        self.data = df["return"].fillna(0)

    def fit(self, p,q):

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

        # Attach p,q to self. Use: naming models
        self.p=p
        self.q=q

        # Identify the index of 80th percentile
        cutoff_test = int(len(self.data)*0.8)
        
        # Create training dataset 
        train_data = self.data.iloc[:cutoff_test]
        
        # Fit a GARCH model
        model = arch_model(
            train_data, 
            p=p,
            q=q, 
            rescale = False
        ).fit(disp=0)
        
        # Train Model, attach to `self.model`
        self.model = model
        self.aic = model.aic
        self.bic = model.bic

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


    def dump(self):

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
        filepath = os.path.join(self.model_directory, f"{timestamp}-ARCH{self.p}{self.q}_{self.ticker}.pkl")
        
        # Save `self.model`
        joblib.dump(self.model, filepath)

        # Return filepath
        return filepath
    

    def load(self, use_model):

        """Load most recent model in `self.model_directory` for `self.ticker`,
        attach to `self.model` attribute.

        """
        # Create pattern for glob search
        pattern = os.path.join(settings.model_directory, f"*{self.ticker}.pkl")
        # Use glob to get most recent model, handle errors
        try:
            if use_model == "latest":
                model_path = sorted(glob(pattern))[-1]
            else:
                model_path = os.path.join(settings.model_directory, f"{use_model}.pkl")
        except IndexError:
            raise Exception(f"No model trained for '{self.ticker}'")
        
        # Load model and attach to `self.model`
        self.model = joblib.load(model_path)
        self.model_name = model_path

        return self.model

        
    