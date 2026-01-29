# Car Purchase Amount — ANN Regression

Predict car purchase amount from customer features using a feedforward Artificial Neural Network (TensorFlow/Keras).

## Dataset

- **File:** `car_purchasing.csv`
- **Target:** `car purchase amount` (regression)
- **Features:** gender, age, annual Salary, credit card debt, net worth

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Open and run `car_purchasing_ann.ipynb` in Jupyter or Cursor. The notebook:

1. Loads and explores the data
2. Preprocesses features and target (StandardScaler)
3. Builds and trains an ANN (64 → 32 → 1)
4. Evaluates on test set (MSE, MAE, actual vs predicted plot)

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn, tensorflow
