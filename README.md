# Wine Quality API

This is a machine learning API built with FastAPI that predicts wine quality based on its physiochemical properties. The model used is a pre-trained Random Forest Regressor/Classifier.

## Dataset
The model is trained on the Red Wine Quality dataset. It includes features like:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

## Project Structure
- `main.py`: The FastAPI application that exposes the prediction endpoint.
- `utils.py`: Contains the data preprocessing pipeline to prepare new data for the model.
- `notebook.ipynb`: Jupyter notebook for exploratory data analysis and model training.
- `Model_RandomForest.pkl`: The serialized Random Forest model.
- `winequality-red.csv`: The dataset used to train the model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedEsam98/wine-quality-api.git
   cd "wine-quality-api"
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the API server locally:
```bash
uvicorn main:app --reload
```

Then, you can access the interactive API documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Make a Prediction

The API has a GET endpoint `/root` which takes the physiochemical properties as query parameters. You can make a request to test it from the Swagger UI or using `curl`:

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/root?fixed_acidity=7.4&volatile_acidity=0.7&citric_acid=0&residual_sugar=1.9&chlorides=0.076&free_sulfur_dioxide=11&total_sulfur_dioxide=34&density=0.9978&pH=3.51&sulphates=0.56&alcohol=9.4' \
  -H 'accept: application/json'
```

## Technologies Used
- Python
- FastAPI
- Scikit-Learn
- Pandas
- Numpy
- Joblib
