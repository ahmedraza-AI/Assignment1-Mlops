from tensorflow.keras.models import load_model
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




model = load_model('iris_model.h5')

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

np.savetxt("iris_training_data.csv", np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1), delimiter=",")
np.savetxt("iris_testing_data.csv", np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1), delimiter=",")
