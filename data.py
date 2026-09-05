"""This is for all the code used to interact with the AlphaVantage API
and the SQLite database. Remember that the API relies on a key that is
stored in the `.env` file and imported via the `config` module.
"""

import pandas as pd
import requests
from config import settings
from datetime import datetime, timedelta, timezone

def get_start_date(period):
    lookback = {
        '1y' : 1,
        '3y' : 3,
        '5y' : 5
    }
    
    latestDate = get_latest_expected_eod()
    startDate = (latestDate - timedelta(days=365*lookback[period])).strftime("%Y-%m-%d")
    return startDate

def get_latest_expected_eod() -> pd.Timestamp:
    now_tz = pd.Timestamp.now(tz=settings.exchange_tz)
    
    # If it's before 5:00 PM Eastern, the latest completed EOD bar is from the previous business day
    if now_tz.hour < settings.eod_available_hour:
        expected = now_tz - pd.tseries.offsets.BDay(1)
    else:
        expected = now_tz
        
    # If the computed date falls on a weekend, roll back to Friday
    if expected.dayofweek > 4:
        expected -= pd.tseries.offsets.BDay(1)
        
    # Return as timezone-naive midnight timestamp to match SQLite date formats
    return expected.normalize().tz_localize(None)

def get_current_timestamp():
    timestamp = datetime.now(timezone.utc)
    return timestamp


class TwelveDataAPI:
    def __init__(self):
        self.__api_key = settings.twelve_data_api_key

    def fetch_data_from_api(self, identifier, start_date, idType="ticker"):
    
        """Get daily time series of an equity from Twelve Data API.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the equity.
        period : str, optional
            Lookback period for the time series data. Options are "1y",
            "3y", and "5y". By default "5y".
        interval : str, optional
            Time interval between two consecutive data points.
            By default "1day".

        Returns
        -------
        pd.DataFrame
            Columns are 'open', 'high', 'low', 'close', and 'volume'.
            All columns are numeric.
        """
        
        url = (
            "https://api.twelvedata.com/time_series?"
            f"{'symbol' if idType == 'ticker' else 'isin'}={identifier}&"
            f"interval=1day&"
            f"start_date={start_date}&"
            f"apikey={self.__api_key}"
        )
        
        
        # Send request to API
        response = requests.get(url)
        response_data = response.json() #returns a json with two keys: meta and values

        # Error handling: if API call was unsuccessful, raise exception with error message
        if "meta" not in response_data.keys():
            error_msg = response_data['message']
            raise Exception(
                f"Invalid API call. Error message: {error_msg}"
            )

        # Convert API response to DataFrame
        df = pd.DataFrame(response_data['values'])

        # Set 'datetime' column as index
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        
        # Convert 'open', 'high', 'low', 'close' to float 
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float) 

        # Convert 'volume' to integer
        df['volume'] = df['volume'].astype(int) 
        
        # Sort index from latest to oldest
        df.sort_index(ascending=False, inplace=True)

        # Return results
        return df


class SQLRepository:
    def __init__(self, connection):
        self.connection = connection

    def insert_table(self, table_name, records, if_exists="replace"):
    
        """Insert DataFrame into SQLite database as table

        Parameters
        ----------
        table_name : str
        records : pd.DataFrame
        if_exists : str, optional
            How to behave if the table already exists.

            - 'fail': Raise a ValueError
            - 'replace': Drop the table before inserting new values.
            - 'append': Insert new values to the existing table.

            Dafault: 'fail'

        Returns
        -------
        dict
            Dictionary has two keys:

            - 'transaction_successful', followed by bool
            - 'records_inserted', followed by int
        """
        
        n_inserted = records.to_sql(name=table_name, con=self.connection, if_exists=if_exists)
        
        return {
            'transaction_successful':True, 'records_inserted':n_inserted
        }


    def read_table(self, table_name: str, date_start: str, date_end: str) -> pd.DataFrame:
    
        """
        Read table from database.
        """
        import sqlite3
        
        query = f'''
        SELECT * 
        FROM {table_name}
        WHERE DATE(date) >= DATE(?) 
            AND DATE(date) <= DATE(?)
        ORDER BY date ASC
        '''
         
        try:
            df = pd.read_sql(query, self.connection, params=(date_start, date_end), index_col='date', parse_dates=['date'])
        except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
            if f"no such table: {table_name}" in str(exc):
                raise ValueError(f"Table '{table_name}' does not exist.") from exc
            raise
             
        if df.empty:
            raise ValueError(f"Table '{table_name}' contains no data for the specified date range.")
            
        # Return DataFrame
        return df
