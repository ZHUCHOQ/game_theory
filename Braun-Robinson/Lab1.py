import numpy as np

# Матрица игры
A = np.array([[1, 17, 18],
              [14, 6, 16],
              [14, 14, 13]])

def monotonous_iterative(A):
    # Начало итерации
    x = np.zeros(len(A))
    x[0] = 1
    c = A[0]
    
    for _ in range(1000):  # Ограничиваем число итераций
        J = np.argwhere(c == np.min(c)).flatten()
        
        if len(J) == 1:
            x_new = np.zeros_like(x)
            x_new[J[0]] = 1
        else:
            # Нахождение оптимального распределения внутри индекса минимума
            # Используется упрощённый подход, поскольку в реальной задаче потребуется решать подигру
            x_new = np.random.dirichlet(np.ones(len(J)))
            
        alpha = 0.5  # Шаг итерации
        x = (1-alpha)*x + alpha*x_new
        c = (1-alpha)*c + alpha*A.dot(x)
        
        # Критерий останова
        if np.isclose(alpha, 0):
            break
    
    return x, np.min(c)

solution_x, solution_v = monotonous_iterative(A)
print("Оптимальная стратегия первого игрока:", solution_x)
print("Цена игры:", solution_v)