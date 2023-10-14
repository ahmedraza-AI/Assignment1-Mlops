from tensorflow.keras.models import load_model
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pytest

@pytest.fixture
def load_iris_model():
    model = load_model('iris_model.h5')
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_test, y_test, model
    
def test_model_accuracy(load_iris_model):
    X_test, y_test, model = load_iris_model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
    assert accuracy >= 0.95, "The model is below the accuracy mark!"

