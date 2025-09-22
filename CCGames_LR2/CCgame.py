import numpy as np

def H(x, y, a, b, c, d, e):
    """Функция выигрыша (ядро игры)"""
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y

def check_saddle_point(A):
    """Проверка наличия седловой точки в матрице"""
    row_min = np.min(A, axis=1)
    col_max = np.max(A, axis=0)
    saddle_value = np.max(row_min)
    if saddle_value == np.min(col_max):
        saddle_point = (np.argmax(row_min), np.argmin(col_max))
        return True, saddle_point, saddle_value
    return False, None, None

def brown_robinson(A, K=1000):
    """Метод Брауна-Робинсона для поиска смешанных стратегий"""
    m, n = A.shape
    accum_row = np.zeros(m)
    accum_col = np.zeros(n)
    p = np.zeros(m)
    q = np.zeros(n)
    
    for k in range(K):
        i_k = np.argmax(accum_row)
        j_k = np.argmin(accum_col)
        p[i_k] += 1
        q[j_k] += 1
        accum_row += A[:, j_k]
        accum_col += A[i_k, :]
    
    v_upper = np.max(accum_row) / K
    v_lower = np.min(accum_col) / K
    price = (v_upper + v_lower) / 2
    x_strategy = p / K
    y_strategy = q / K
    return x_strategy, y_strategy, price

def main():
    # Параметры для варианта 9
    a, b, c, d, e = -6, 32/5, 16, -16/5, -64/5
    results = {}
    
    for N in range(2, 12):  # N от 2 до 11
        # Создаем сетку
        x_vals = np.linspace(0, 1, N+1)
        y_vals = np.linspace(0, 1, N+1)
        
        # Создаем матрицу выигрышей
        A = np.zeros((N+1, N+1))
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                A[i, j] = H(x, y, a, b, c, d, e)
        
        print(f"\nN = {N}")
        print("Матрица выигрышей:")
        print(np.round(A, 4))
        
        # Проверяем наличие седловой точки
        has_saddle, saddle_point, saddle_value = check_saddle_point(A)
        
        if has_saddle:
            x_index, y_index = saddle_point
            x_opt = x_vals[x_index]
            y_opt = y_vals[y_index]
            results[N] = ('Saddle', x_opt, y_opt, saddle_value)
            print(f"Седловая точка: x={x_opt:.3f}, y={y_opt:.3f}, H={saddle_value:.6f}")
        else:
            # Используем метод Брауна-Робинсона
            x_strategy, y_strategy, price = brown_robinson(A)
            x_opt = np.dot(x_vals, x_strategy)
            y_opt = np.dot(y_vals, y_strategy)
            results[N] = ('Brown-Robinson', x_opt, y_opt, price)
            print(f"Метод Брауна-Робинсона: x={x_opt:.3f}, y={y_opt:.3f}, H={price:.6f}")
    
    print("\nИтоговые результаты для N=2 до 11:")
    for N, (method, x, y, value) in results.items():
        print(f"N={N}: {method}, x={x:.3f}, y={y:.3f}, H={value:.6f}")
    
    # Сравнение с аналитическим решением
    analytical_value = -3.84
    numerical_value = results[11][3]  # Результат для N=11
    error = abs(analytical_value - numerical_value)
    print(f"\nАналитическое значение: {analytical_value}")
    print(f"Численное значение для N=11: {numerical_value}")
    print(f"Погрешность: {error:.10f}")

if __name__ == "__main__":
    main()