# Predicting Stock Volatility using Python

## Introduction

This project demonstrates how to predict stock volatility using Python. It pulls historical stock prices from the TwelveData API, stores the data in a SQLite database, trains an ARCH (Autoregressive Conditional Heteroskedasticity) model for volatility prediction, and deploys the model via a FastAPI web service. This work is inspired by a project from the WorldQuant Applied Data Science program.

The project is designed for data science enthusiasts interested in financial modeling, time series analysis, and machine learning deployment. It assumes basic familiarity with Python, but newcomers will find guidance below.

### Prerequisites

- **Python**: If you're new to Python, it's a versatile programming language. Install it from [python.org](https://www.python.org/downloads/). We recommend Python 3.8 or later.
- **Docker**: A tool to run applications in containers. Install from [docker.com](https://www.docker.com/get-started). It's essential for running this project easily without worrying about dependencies.
- **Jupyter Notebooks**: For interactive demos. If new, think of it as a web-based environment to run Python code in cells. It's included in the Docker setup.

## API Guide

### `POST /fit`

Description: Trains an ARCH model on historical data and saves it in the `/models` folder.

Parameters:
- `ticker`: A stock ticker symbol (e.g., "AAPL").
- `n_observations`: Number of past data points to use for training (integer).
- `p`, `q`: ARCH model parameters (integers).

Returns: A JSON message with the name of the trained model.

### `POST /predict`

Description: Uses a trained model to predict future volatility.

Parameters:
- `ticker`: A stock ticker symbol.
- `n_days`: Number of days to predict ahead (integer).
- `use_model`: Model to use ("latest" for the most recent model in `/models`).

Returns: Predicted volatility values as JSON.

## How to Run the Project with Docker Build

1. **Clone the Repository**: Download the project files from GitHub to your local machine.
2. **Navigate to the Project Directory**: Open a terminal and go to the folder containing the project (e.g., `cd /d:/1 PROJECTS/Github/predicting-stock-volatility-using-Python`).
3. **Build the Docker Image**: Run `docker build .` to create a container image with all dependencies.
4. **Run the Container**: Execute `docker run -p 8000:8000 <image_name>` (replace `<image_name>` with the built image ID or name).
5. **Access Swagger UI**: Open a web browser and go to `http://localhost:8000/docs` to interact with the API endpoints via Swagger. This is a user-friendly interface to test the `/fit` and `/predict` endpoints without coding.

If you're new to Docker, it packages the app and its environment, so you don't need to install Python libraries manually.

## How to Run the Project and Demo with Docker Compose

Docker Compose simplifies running multiple services together, including the API and a Jupyter Notebook for the demo.

1. **Ensure Docker Compose is Installed**: It comes with Docker Desktop. If not, follow [Docker's guide](https://docs.docker.com/compose/install/).
2. **Clone the Repository**: As above.
3. **Navigate to the Directory**: As above.
4. **Run with Compose**: Execute `docker-compose up` (or `docker compose up` on newer versions). This starts the FastAPI service and a Jupyter Notebook server.
5. **Access the API**: Visit `http://localhost:8000/docs` for Swagger.
6. **Access the Demo Notebook**: Open `http://localhost:8888` in your browser. If prompted, enter the token from the terminal output (usually shown when Compose starts). The notebook (`project_demo.ipynb`) walks through data fetching, model training, and predictions interactively.

For Jupyter newcomers: Notebooks run code in cells. Click a cell and press Shift+Enter to execute. Explore the demo to see volatility predictions in action. If issues arise, check Docker logs with `docker-compose logs`.

## How to Stop and Close Containers

To stop and remove running containers:

- **For Docker Build**: In the terminal, run `docker ps` to list running containers. Note the container ID, then run `docker stop <container_id>` followed by `docker rm <container_id>` to stop and remove it.
- **For Docker Compose**: In the project directory, run `docker-compose down` (or `docker compose down` on newer versions) to stop and remove all services defined in the compose file.

This ensures resources are freed and prevents conflicts on subsequent runs.

## Project Demo

The demo notebook provides a step-by-step guide to the project's workflow, including data visualization and model evaluation.
