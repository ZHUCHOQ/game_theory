import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ===== 1. НОРМАЛИЗАЦИЯ МАТРИЦЫ =====
print("1. НОРМАЛИЗАЦИЯ МАТРИЦЫ")
print("=" * 50)

# Исходная матрица для варианта 9
A = np.array([
    [-5, -1, -1, -3, 0],
    [-4, 1, 0, -4, 0],
    [-6, -1, 0, -5, 0],
    [-3, 0, 0, -3, 0],
    [2, -3, -4, 3, -3]
])

print("Исходная матрица:")
print(A)

# Находим минимальный элемент
min_element = np.min(A)
k = -min_element  # Константа для нормализации
print(f"\nМинимальный элемент: {min_element}")
print(f"Константа нормализации k = {k}")

# Нормализуем матрицу
A_norm = A + k
print("\nНормализованная матрица:")
print(A_norm)

# ===== 2. СВЕДЕНИЕ К ИГРЕ 2×2 ПУТЕМ ПОГЛОЩЕНИЯ ДОМИНИРУЕМЫХ СТРАТЕГИЙ =====
print("\n2. СВЕДЕНИЕ К ИГРЕ 2×2 ПУТЕМ ПОГЛОЩЕНИЯ ДОМИНИРУЕМЫХ СТРАТЕГИЙ")
print("=" * 80)

def remove_dominated_strategies(matrix):
    """Удаляет доминируемые строки и столбцы"""
    m, n = matrix.shape
    rows_to_keep = list(range(m))
    cols_to_keep = list(range(n))
    
    # Удаление доминируемых строк
    i = 0
    while i < len(rows_to_keep):
        j = 0
        while j < len(rows_to_keep):
            if i != j:
                # Проверяем, доминирует ли строка j строку i
                if all(matrix[rows_to_keep[j], k] >= matrix[rows_to_keep[i], k] for k in range(n)):
                    # Строка j доминирует строку i - удаляем строку i
                    rows_to_keep.pop(i)
                    break
            j += 1
        else:
            i += 1
    
    # Удаление доминируемых столбцов
    i = 0
    while i < len(cols_to_keep):
        j = 0
        while j < len(cols_to_keep):
            if i != j:
                # Проверяем, доминирует ли столбец j столбец i
                if all(matrix[k, cols_to_keep[j]] <= matrix[k, cols_to_keep[i]] for k in range(m)):
                    # Столбец j доминирует столбец i - удаляем столбец i
                    cols_to_keep.pop(i)
                    break
            j += 1
        else:
            i += 1
    
    return matrix[np.ix_(rows_to_keep, cols_to_keep)], rows_to_keep, cols_to_keep

# Применяем алгоритм к нормализованной матрице
B, kept_rows, kept_cols = remove_dominated_strategies(A_norm)
print("Матрица после удаления доминируемых стратегий:")
print(B)
print(f"Сохранились строки: {kept_rows}")
print(f"Сохранились столбцы: {kept_cols}")

# ===== 3. СВЕДЕНИЕ К ИГРЕ 2×2 ПУТЕМ УДАЛЕНИЯ NBR-СТРАТЕГИЙ =====
print("\n3. СВЕДЕНИЕ К ИГРЕ 2×2 ПУТЕМ УДАЛЕНИЯ NBR-СТРАТЕГИЙ")
print("=" * 80)

def is_best_response(player, strategy, matrix, opponent_strategy):
    """Проверяет, является ли стратегия наилучшим ответом"""
    if player == 1:  # Игрок 1 (строки)
        payoffs = matrix @ opponent_strategy
        return np.argmax(payoffs) == strategy
    else:  # Игрок 2 (столбцы)
        payoffs = strategy @ matrix
        return np.argmin(payoffs) == strategy

def remove_nbr_strategies(matrix):
    """Удаляет стратегии, которые никогда не являются наилучшим ответом"""
    m, n = matrix.shape
    active_rows = list(range(m))
    active_cols = list(range(n))
    changed = True
    
    while changed and (len(active_rows) > 2 or len(active_cols) > 2):
        changed = False
        
        # Проверяем строки
        for i in active_rows[:]:
            is_nbr = True
            # Генерируем случайные смешанные стратегии для игрока 2
            for _ in range(100):
                mixed_strategy = np.random.dirichlet(np.ones(len(active_cols)))
                if is_best_response(1, i, matrix[np.ix_(active_rows, active_cols)], mixed_strategy):
                    is_nbr = False
                    break
            
            if is_nbr and len(active_rows) > 2:
                active_rows.remove(i)
                changed = True
                break
        
        # Проверяем столбцы
        for j in active_cols[:]:
            is_nbr = True
            # Генерируем случайные смешанные стратегии для игрока 1
            for _ in range(100):
                mixed_strategy = np.random.dirichlet(np.ones(len(active_rows)))
                if is_best_response(2, j, mixed_strategy, matrix[np.ix_(active_rows, active_cols)]):
                    is_nbr = False
                    break
            
            if is_nbr and len(active_cols) > 2:
                active_cols.remove(j)
                changed = True
                break
    
    return matrix[np.ix_(active_rows, active_cols)], active_rows, active_cols

# Применяем алгоритм NBR
B_nbr, nbr_rows, nbr_cols = remove_nbr_strategies(A_norm)
print("Матрица после удаления NBR-стратегий:")
print(B_nbr)
print(f"Сохранились строки: {nbr_rows}")
print(f"Сохранились столбцы: {nbr_cols}")

# ===== 4. ГРАФОАНАЛИТИЧЕСКИЙ МЕТОД =====
print("\n4. ГРАФОАНАЛИТИЧЕСКИЙ МЕТОД")
print("=" * 50)

# Используем матрицу 2x2, полученную ранее
if B.shape == (2, 2):
    print("Матрица для анализа:")
    print(B)
    
    # Решение для игрока 1
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]
    
    # Находим оптимальную стратегию игрока 1
    p = (d - c) / (a + d - b - c)
    v = (a*d - b*c) / (a + d - b - c)
    
    # Находим оптимальную стратегию игрока 2
    q = (d - b) / (a + d - b - c)
    
    print(f"Оптимальная стратегия игрока 1: p = {p:.3f}")
    print(f"Оптимальная стратегия игрока 2: q = {q:.3f}")
    print(f"Цена игры (нормализованная): v = {v:.3f}")
    
    # Графическое представление
    p_values = np.linspace(0, 1, 100)
    E1 = a * p_values + c * (1 - p_values)  # Ожидаемый выигрыш против столбца 1
    E2 = b * p_values + d * (1 - p_values)  # Ожидаемый выигрыш против столбца 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, E1, 'b-', label='Против столбца 1')
    plt.plot(p_values, E2, 'r-', label='Против столбца 2')
    plt.axhline(y=v, color='g', linestyle='--', label=f'Цена игры v = {v:.3f}')
    plt.axvline(x=p, color='m', linestyle='--', label=f'Оптимальная p = {p:.3f}')
    plt.xlabel('Вероятность выбора первой строки (p)')
    plt.ylabel('Ожидаемый выигрыш')
    plt.title('Графоаналитический метод: поиск оптимальной стратегии')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Матрица не 2x2, графический метод не применим")

# ===== 5. АНАЛИТИЧЕСКИЙ (МАТРИЧНЫЙ) МЕТОД =====
print("\n5. АНАЛИТИЧЕСКИЙ (МАТРИЧНЫЙ) МЕТОД")
print("=" * 50)

if B.shape == (2, 2):
    # Формулы для матричной игры 2x2
    det = a*d - b*c
    trace = a + d - b - c
    
    p_matrix = (d - c) / trace
    q_matrix = (d - b) / trace
    v_matrix = det / trace
    
    print(f"Оптимальная стратегия игрока 1: p = {p_matrix:.3f}")
    print(f"Оптимальная стратегия игрока 2: q = {q_matrix:.3f}")
    print(f"Цена игры (нормализованная): v = {v_matrix:.3f}")

# ===== 6. СИМПЛЕКС-МЕТОД =====
print("\n6. СИМПЛЕКС-МЕТОД")
print("=" * 50)

# Задача линейного программирования для игрока 1 (минимизация максимальных потерь)
if B.shape == (2, 2):
    # Целевая функция: минимизировать v
    c = [0, 0, 1]  # [x1, x2, v]
    
    # Ограничения: B^T * x >= v, x1 + x2 = 1, x1, x2 >= 0
    A_ub = [
        [-B[0, 0], -B[1, 0], 1],  # -a*x1 - c*x2 + v <= 0
        [-B[0, 1], -B[1, 1], 1]   # -b*x1 - d*x2 + v <= 0
    ]
    b_ub = [0, 0]
    
    # Ограничение равенства: x1 + x2 = 1
    A_eq = [[1, 1, 0]]
    b_eq = [1]
    
    # Границы переменных
    bounds = [(0, None), (0, None), (None, None)]
    
    # Решаем задачу ЛП
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    if result.success:
        x1, x2, v_simplex = result.x
        print(f"Оптимальная стратегия игрока 1: x1 = {x1:.3f}, x2 = {x2:.3f}")
        print(f"Цена игры (нормализованная): v = {v_simplex:.3f}")
    else:
        print("Симплекс-метод не смог найти решение")

# ===== 7. РАСЧЕТ ЦЕНЫ ИГРЫ ДЛЯ ИСХОДНОЙ МАТРИЦЫ =====
print("\n7. РАСЧЕТ ЦЕНЫ ИГРЫ ДЛЯ ИСХОДНОЙ МАТРИЦЫ")
print("=" * 50)

# Цена игры для исходной матрицы
v_original = v - k
print(f"Цена игры для нормализованной матрицы: {v:.3f}")
print(f"Константа нормализации: k = {k}")
print(f"Цена игры для исходной матрицы: v = {v:.3f} - {k} = {v_original:.3f}")

# ===== ВЫВОД ОПТИМАЛЬНЫХ СТРАТЕГИЙ =====
print("\nОПТИМАЛЬНЫЕ СМЕШАННЫЕ СТРАТЕГИИ")
print("=" * 50)

# Восстанавливаем стратегии для исходной матрицы 5x5
if B.shape == (2, 2) and len(kept_rows) == 2 and len(kept_cols) == 2:
    # Стратегия игрока 1
    p1_optimal = np.zeros(5)
    p1_optimal[kept_rows[0]] = p
    p1_optimal[kept_rows[1]] = 1 - p
    
    # Стратегия игрока 2
    p2_optimal = np.zeros(5)
    p2_optimal[kept_cols[0]] = q
    p2_optimal[kept_cols[1]] = 1 - q
    
    print("Оптимальная стратегия игрока 1:")
    for i, prob in enumerate(p1_optimal):
        print(f"  Стратегия {i+1}: {prob:.3f}")
    
    print("\nОптимальная стратегия игрока 2:")
    for i, prob in enumerate(p2_optimal):
        print(f"  Стратегия {i+1}: {prob:.3f}")

print(f"\nИтоговая цена игры: {v_original:.3f}")