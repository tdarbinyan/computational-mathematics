import numpy as np

def QR_Decomposition(A):
    n, m = A.shape

    Q = np.empty((n, min(n, m)))
    u = np.empty((n, min(n, m)))

    # Получаем первый нормированный и в будущем ортогональный вектор
    u[:, 0] = A[:, 0] 
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, min(n, m)):

        u[:, i] = A[:, i]
        for j in range(i):
            # Вычитание из u проекции уже полученных "базисных" векторов
            u[:, i] -= (u[:, i] @ Q[:, j]) * Q[:, j] 

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # Нормировка

    R = np.zeros((min(n, m), m))
    
    # for цикл для заполнения верхнетреугольной матрицы
    for i in range(min(n, m)):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i] # заполнение матрицы R

    return Q, R

def back_substitution(U, y): # Метод обратной подстановки

    n = U.shape[0] # Получение количества строк
    x = np.zeros_like(y, dtype=np.double); # Создаем массив для хранения решений

    x[-1] = y[-1] / U[-1, -1] # Считаем значение последнего аргумента(тривиальный случай)

    for i in range(n-2, -1, -1):
        # Идем с конца, вычитаем уже посчитанные значения помножив на коэффициенты, находим очередной элемент
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i] 

    return x

# Тестовые случаи с доски
A = np.array([[2.0, -2.0, 18.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
# A = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])

print("-------------- [  A  ] --------------")
print(A)
print("")

Q, R = QR_Decomposition(A) # QR разложение

print("-------------- [  Q  ] --------------")
print("")
print(Q)
print("")

print("-------------- [  R  ] --------------")
print("")
print(R)
print("")

# Получение вектора решения
x = back_substitution(R, Q.T @ np.array([1.0, 1.0, 1.0]))

# Получение вектора решения с помощью numpy
x_np = np.linalg.solve(A, np.array([1.0, 1.0, 1.0]))

print("-------------- [  x  ] --------------")
print(f"My solution:    {x}") # Собственное решение
print("")
print(f"Numpy solution: {x_np}") # Решение с помощью numpy
print("")
print(f"Difference: {np.linalg.norm(x - x_np)}") # Разность моего и стандартного решения