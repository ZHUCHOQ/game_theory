import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Исходная матрица для варианта 9
original_matrix = np.array([
    [-5, -1, -1, -3, 0],
    [-4, 1, 0, -4, 0],
    [-6, -1, 0, -5, 0],
    [-3, 0, 0, -3, 0],
    [2, -3, -4, 3, -3]
])

print("Исходная матрица:")
print(original_matrix)

# 1. Нормализация матрицы
normalizer = 6
normalized_matrix = original_matrix + normalizer
print("\nНормализованная матрица:")
print(normalized_matrix)

# 2. Уменьшение размерности - корректная реализация
def reduce_matrix(matrix):
    """
    Последовательно применяем доминирование строк и столбцов
    """
    reduced = matrix.copy()
    changed = True
    
    while changed:
        changed = False
        m, n = reduced.shape
        
        # Удаление доминируемых строк
        rows_to_remove = []
        for i in range(m):
            for j in range(m):
                if i != j and i not in rows_to_remove and j not in rows_to_remove:
                    # Строка i доминирует строку j если все элементы i >= соответствующим элементам j
                    if all(reduced[i] >= reduced[j]):
                        rows_to_remove.append(j)
                        changed = True
        
        # Удаление доминируемых столбцов
        cols_to_remove = []
        for i in range(n):
            for j in range(n):
                if i != j and i not in cols_to_remove and j not in cols_to_remove:
                    # Столбец i доминирует столбец j если все элементы i <= соответствующим элементам j
                    if all(reduced[:, i] <= reduced[:, j]):
                        cols_to_remove.append(j)
                        changed = True
        
        # Удаляем отмеченные строки и столбцы
        if rows_to_remove or cols_to_remove:
            rows_to_keep = [i for i in range(m) if i not in rows_to_remove]
            cols_to_keep = [i for i in range(n) if i not in cols_to_remove]
            reduced = reduced[np.ix_(rows_to_keep, cols_to_keep)]
    
    return reduced

# Применяем сокращение
reduced_matrix = reduce_matrix(normalized_matrix)
print("\nМатрица после сокращения доминированием:")
print(reduced_matrix)

# Если матрица не 2x2, выбираем подматрицу 2x2 для демонстрации
if reduced_matrix.shape != (2, 2):
    print("\nМатрица после сокращения не 2x2, выбираем подматрицу 2x2 для примера")
    # Выбираем первые 2 строки и 2 столбца
    final_matrix = reduced_matrix[:2, :2]
else:
    final_matrix = reduced_matrix

print("\nФинальная матрица 2x2 для решения:")
print(final_matrix)

# 3. Решение игры 2×2 аналитическим методом
def solve_2x2_game(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Проверка на особый случай
    if (a + d - b - c) == 0:
        print("Особый случай: знаменатель равен 0")
        return None, None, None
    
    # Цена игры
    v = (a*d - b*c) / (a + d - b - c)
    
    # Смешанная стратегия игрока A
    p = (d - c) / (a + d - b - c)
    q = 1 - p
    
    # Смешанная стратегия игрока B
    r = (d - b) / (a + d - b - c)
    s = 1 - r
    
    return v, (p, q), (r, s)

game_price, strategy_A, strategy_B = solve_2x2_game(final_matrix)
print(f"\nАналитическое решение:")
print(f"Цена игры: {game_price:.3f}")
print(f"Стратегия игрока A: ({strategy_A[0]:.3f}, {strategy_A[1]:.3f})")
print(f"Стратегия игрока B: ({strategy_B[0]:.3f}, {strategy_B[1]:.3f})")

# 4. Графоаналитический метод
def graphical_solution(matrix):
    x = np.linspace(0, 1, 100)
    
    # Ожидаемые выигрыши для стратегий игрока B
    payoff_b1 = matrix[0,0] * x + matrix[1,0] * (1 - x)
    payoff_b2 = matrix[0,1] * x + matrix[1,1] * (1 - x)
    
    # Находим точку пересечения
    idx = np.argmin(np.abs(payoff_b1 - payoff_b2))
    x_opt = x[idx]
    v_opt = payoff_b1[idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, payoff_b1, label='Стратегия B1')
    plt.plot(x, payoff_b2, label='Стратегия B2')
    plt.axvline(x=x_opt, color='r', linestyle='--', label=f'Оптимум: x={x_opt:.3f}')
    plt.axhline(y=v_opt, color='g', linestyle='--', label=f'Цена игры: v={v_opt:.3f}')
    plt.xlabel('Вероятность выбора стратегии A1')
    plt.ylabel('Ожидаемый выигрыш')
    plt.legend()
    plt.title('Графоаналитическое решение')
    plt.grid(True)
    plt.show()
    
    return x_opt, v_opt

print("\nГрафоаналитическое решение:")
x_opt, v_opt = graphical_solution(final_matrix)
print(f"Оптимальная стратегия A1: {x_opt:.3f}")
print(f"Цена игры: {v_opt:.3f}")

# 5. Симплекс-метод
def simplex_solution(matrix):
    # Задача для игрока A (минимизация)
    c = [1, 1]  # F = x1 + x2 → min
    A_ub = [[-matrix[0,0], -matrix[1,0]],  # Ограничения ≥ 1
            [-matrix[0,1], -matrix[1,1]]]
    b_ub = [-1, -1]
    bounds = [(0, None), (0, None)]
    
    res_A = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res_A.success:
        game_price_A = 1 / res_A.fun
        strategy_A_simplex = res_A.x / res_A.fun
        return game_price_A, strategy_A_simplex
    else:
        return None, None

game_price_simplex, strategy_simplex = simplex_solution(final_matrix)
print(f"\nСимплекс-метод:")
print(f"Цена игры: {game_price_simplex:.3f}")
print(f"Стратегия игрока A: ({strategy_simplex[0]:.3f}, {strategy_simplex[1]:.3f})")

# 6. Расчет цены игры для исходной матрицы
original_game_price = game_price - normalizer
print(f"\nЦена игры для исходной матрицы: {original_game_price:.3f}")

# Проверка оптимальности стратегий
def check_optimality(matrix, strategy_A, strategy_B, v):
    print("\nПроверка оптимальности стратегий:")
    
    # Ожидаемый выигрыш при оптимальных стратегиях
    expected_payoff = 0
    for i in range(len(strategy_A)):
        for j in range(len(strategy_B)):
            expected_payoff += strategy_A[i] * strategy_B[j] * matrix[i, j]
    
    print(f"Ожидаемый выигрыш: {expected_payoff:.3f}")
    print(f"Цена игры: {v:.3f}")
    print(f"Разница: {abs(expected_payoff - v):.6f}")
    
    return abs(expected_payoff - v) < 1e-6

# Проверяем для нормализованной матрицы
check_optimality(final_matrix, strategy_A, strategy_B, game_price)