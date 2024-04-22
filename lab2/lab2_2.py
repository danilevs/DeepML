import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Загрузка датасета Iris
df = pd.read_csv('data.csv')
X = df.iloc[:, :4]
y = df.iloc[:, 4].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Преобразование меток классов в one-hot encoding
num_classes = 3
y_one_hot = np.eye(num_classes)[y]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=1)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = X_train.shape[1]
output_size = num_classes
weights = np.random.rand(input_size, output_size)
bias = np.zeros((1, output_size))


def predict(X):
    return np.dot(X, weights) + bias

def train(X, y, learning_rate=0.01, epochs=100):
    global weights
    global bias
    for epoch in range(epochs):
        if epoch%10==0:
            print("Epoch:", epoch, "/", epochs)
        for i in range(X.shape[0]):
            input_data = X[i, :].reshape(1, -1)
            target = y[i, :].reshape(1, -1)

           # Прямой проход (Forward pass)
            output = predict(input_data)

           # Обратный проход (обновление весов и смещения)
            weights += learning_rate * np.dot(input_data.T, (target - output))
            bias += learning_rate * (target - output)

# Создаем и обучаем перцептрон
train(X_train, y_train)

# Предсказываем классы для тестовых данных
predictions = predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

#Точность
count=0
for i in range(len(y_test)):
    if predicted_classes[i]==np.argmax(y_test, axis=1)[i]:
        count+=1
print(count/len(y_test))
              

