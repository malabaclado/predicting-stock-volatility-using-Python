# Predicting Stock Volatility using Python

Note: Page still under construction.

In this project, I created a program that pulls historical stock prices data from AlphaVantageAPI, store it in a sqlite database, train an ARCH model to predict volatility and deploy the model via FastAPI. This is inspired by my project in WorldQuant Applied Data Science program.

## API guide

### `POST/fit`

Description: This trains an ARCH model and save it in the /models folder.

Parameters:
- `ticker`: a stock ticker
- `n_observations`: the number of past observations to train the model
- `p`,`q`: parameters of the ARCH model

Returns: JSON message with name of the trained model.

### `POST/predict` 

Gets predictions from the model.

Parameters:
- ticker: a stock ticker
- n_days: number of days to predict
- use_model: which model to use ("latest" for the latest model in the /models folder)

Returns: Predicted volatility values

## Project Demo
