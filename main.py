import sqlite3

from config import settings
from data import SQLRepository
from fastapi import FastAPI
from model import GarchModel
from pydantic import BaseModel

# Define input-output classes for `"/fit"` and `"/predict"` paths

class FitIn(BaseModel):
    ticker: str
    n_observations: int
    p: int
    q: int
    

class FitOut(FitIn):
    success: bool
    message: str
    

class PredictIn(BaseModel):
    ticker: str
    n_days: int
    use_model:str = "latest"


class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


# Build model function
def build_model(ticker, use_new_data):
    '''
    
    '''

    # Create DB connection
    connection = sqlite3.connect(database=settings.db_name, check_same_thread=False)

    # Create `SQLRepository`
    repo = SQLRepository(connection)

    # Create model
    model = GarchModel(ticker, repo, use_new_data=use_new_data)

    # Return model
    return model


# Start FastAPI application
app = FastAPI()

# `"/hello" path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    """Return dictionary with greeting message."""
    return {"message" : "Hello, World"}



# `"/fit" path, 200 status code
@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn):

    """Pulls data from TwelveData API, trains GARCH model, and saves model to file.

    Parameters
    ----------
    request : FitIn
        An instance of the `FitIn` class with the following attributes:

    - `ticker`: str, ticker symbol of the equity whose volatility will be predicted.

    - `n_observations`: int, number of observations to retrieve from database for training model.

    - `p`: int, order of GARCH terms in model.
    
    - `q`: int, order of ARCH terms in model.

    Returns
    ------
    dict
        A dictionary with the following keys:

    - `success`: bool, whether model was successfully trained and saved.

    - `message`: str, message with either the filename of the saved model or an error message.

    """
    # Create `response` dictionary from `request`
    response = request.dict()

    # Create try block to handle exceptions
    try:
        # Build model with `build_model` function
        model = build_model(ticker=request.ticker, use_new_data=True)

        # Wrangle data
        model.wrangle_data(n_observations=request.n_observations)

        # Fit model
        model.fit(p=request.p, q=request.q)

        # Save model
        filename = model.dump()

        # Add `"success"` key to `response`
        response["success"] = True


        # Add `"message"` key to `response` with `filename`
        # response["message"] = f"Trained and saved '{filename}'."
        response["message"] = f"Trained and saved '{filename}'. Metrics: AIC {model.aic}, BIC {model.bic}."

    # Create except block
    except Exception as e:
        # Add `"success"` key to `response`
        response["success"] = False

        # Add `"message"` key to `response` with error message
        response["message"] = str(e)

    # Return response
    return response


# `"/predict" path, 200 status code
@app.post("/predict", status_code=200, response_model=PredictOut)
def get_prediction(request: PredictIn):
    """
    Generates volatility predictions for a given stock using a trained GARCH model.

    Parameters
    ----------
    request : PredictIn
        An instance of the `PredictIn` class with the following attributes:

    - `ticker`: str, ticker symbol of the equity whose volatility will be predicted.

    - `n_days`: int, number of days for which to generate predictions.

    - `use_model`: str, name of the stored model to use for predictions.

    Returns
    -------
    dict
        A dictionary with the following keys:

    - `success`: bool, whether prediction was successful.

    - `forecast`: dict, dictionary containing the predicted volatility values.
    
    - `message`: str, message with either the forecast or an error message.
    """

    # Create `response` dictionary from `request`
    response = request.dict()

    # Create try block to handle exceptions
    try:
        # Build model with `build_model` function
        model = build_model(ticker=request.ticker, use_new_data=False)

        # Load stored model
        model.load(request.use_model)

        # Generate prediction
        prediction = model.predict_volatility(horizon = request.n_days)

        # Add `"success"` key to `response`
        response["success"] = True

        # Add `"model_name"` key to `response`
        response["model_name"] = model.model_name

        # Add `"forecast"` key to `response`
        response["forecast"] = prediction

        # Add `"message"` key to `response`
        response["message"] = ""

        

    # Create except block
    except Exception as e:
        # Add `"success"` key to `response`
        response["success"] = False

        # Add `"forecast"` key to `response`
        response["forecast"] = {}

        #  Add `"message"` key to `response`
        response["message"] = str(e)

    # Return response
    return response
