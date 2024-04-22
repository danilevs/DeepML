# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import os

# Проверка наличия файла
data_file_path = 'data.csv'
if not os.path.isfile(data_file_path):
    print("Файл 'data.csv' не найден.")

# Загрузка данных
df = pd.read_csv(data_file_path)

print(df.head())

# Перемешивание
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Выходные данные
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)  # если setosa, то 1

# Входные данные
X = df.iloc[0:100, [0, 2]].values
print(X.shape[1])

# Параметры Персептрона
input_size = X.shape[1]
neurons_hidden_layer = 10
neurons_output_layer = 1


# Инициализация 1 слоя
Win = np.zeros((1 + input_size, neurons_hidden_layer))

Win[0, :] = np.random.randint(0, 3, size=(neurons_hidden_layer))  # пороги
Win[1:, :] = np.random.randint(-1, 2, size=(input_size, neurons_hidden_layer))  # веса

# Инициализация 2 слоя
Wout = np.random.randint(0, 2, size=(1 + neurons_hidden_layer, neurons_output_layer)).astype(np.float64)


# Функция прогнозирования
def predict(X):
    hidden_predict = np.where((np.dot(X, Win[1:, :]) + Win[0, :]) >= 0.0, 1, -1).astype(np.float64)
    Out = np.where((np.dot(hidden_predict, Wout[1:, :]) + Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
    return Out, hidden_predict

# Параметры обучения
n_iter = 0
step = 0.01
check_iter = 5

# Список для хранения матрицы весов второго слоя
list_Wout_weights = []

# Обучение
while True:
    n_iter += 1

    # Обновление весов для каждого образца в обучающем наборе
    for x_input, expected in zip(X, y):
        Out, hidden_predict = predict(x_input)
        Wout[1:] += (step * (expected - Out)) * hidden_predict.reshape(-1, 1)  # Обновление порогов
        Wout[0] += step * (expected - Out)  # Обновление порогового значения

    # Сохранение текущих весов для проверки зацикливания
    list_Wout_weights.append(Wout)
    #print(list_Wout_weights)
    
    # Проверка наличия ошибок
    Out, _ = predict(X)
    sum_errors = sum(Out.reshape(-1) - y)
    if sum_errors == 0:
        print('Все примеры обучающей выборки решены:')
        break

    # Проверка зацикливания каждые check_iter итераций
    if n_iter % check_iter == 0:
        break_out_flag = False
        for item in list_Wout_weights:
            if list_Wout_weights.count(item) > 1:
                print('Повторение весов')
                break_out_flag = True
                break
        if break_out_flag:
            break

# Подсчет ошибок после завершения обучения
Out, _ = predict(X)
sum_errors = sum(Out - y.reshape(-1,1))
print('sum_errors', sum_errors)
print(n_iter)
