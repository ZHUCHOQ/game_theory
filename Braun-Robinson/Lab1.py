import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import deque

# Платежная матрица для варианта 9
C = np.array([
    [19, 7, 3],
    [6, 9, 9],
    [8, 2, 11]
])

# ========== АНАЛИТИЧЕСКИЙ МЕТОД (корректный) ==========
print("АНАЛИТИЧЕСКИЙ МЕТОД (ОБРАТНОЙ МАТРИЦЫ)")
print("=" * 50)

C_inv = np.linalg.inv(C)
u = np.ones(C.shape[0])
uC_inv = u @ C_inv
uC_inv_uT = u @ C_inv @ u.T
v_analytic = 1 / uC_inv_uT
x_analytic = v_analytic * uC_inv
y_analytic = v_analytic * (C_inv @ u.T)

print(f"Цена игры v = {v_analytic:.6f}")
print(f"Оптимальная стратегия игрока A: {x_analytic}")
print(f"Оптимальная стратегия игрока B: {y_analytic}")

# ========== ИСПРАВЛЕННЫЙ МЕТОД БРАУНА-РОБИНСОНА ==========
print("\n" + "=" * 50)
print("ИСПРАВЛЕННЫЙ МЕТОД БРАУНА-РОБИНСОНА")
print("=" * 50)

# Инициализация (как в документе - частоты использования стратегий)
n = 3
x_freq = np.zeros(n)  # Частоты стратегий игрока A (аналогично ˜x_i[k] в документе)
y_freq = np.zeros(n)  # Частоты стратегий игрока B (аналогично ˜y_j[k] в документе)

# Накопленные выигрыши (для вычисления оценок)
A_accumulated = np.zeros(n)  # Накопленные выигрыши для стратегий A
B_accumulated = np.zeros(n)  # Накопленные проигрыши для стратегий B

# История для таблицы
first_10 = []
last_10 = deque(maxlen=10)

max_iterations = 10000

for k in range(1, max_iterations + 1):
    # Выбор стратегий (как в документе)
    if k == 1:
        i_k, j_k = 0, 0  # Произвольный начальный выбор
    else:
        # Игрок A выбирает стратегию, максимизирующую выигрыш против эмпирической стратегии B
        i_k = np.argmax(A_accumulated)
        # Игрок B выбирает стратегию, минимизирующую проигрыш против эмпирической стратегии A  
        j_k = np.argmin(B_accumulated)
    
    # Обновление частот (как в документе: ˜x_i[k] и ˜y_j[k])
    x_freq[i_k] += 1
    y_freq[j_k] += 1
    
    # Обновление накопленных выигрышей (для вычисления оценок)
    for i in range(n):
        A_accumulated[i] += C[i, j_k]  # Выигрыш стратегии i против выбора j_k
    
    for j in range(n):
        B_accumulated[j] += C[i_k, j]  # Проигрыш стратегии j против выбора i_k
    
    # Вычисление оценок цены игры (как в документе)
    upper = np.max(A_accumulated) / k  # ¯v[k]/k
    lower = np.min(B_accumulated) / k  # v[k]/k
    epsilon = upper - lower
    
    # Сохранение истории (частоты использования стратегий, как в документе)
    record = {
        'k': k,
        'choice_A': f'x{i_k + 1}',
        'choice_B': f'y{j_k + 1}',
        'x_freq': x_freq.copy(),  # Частоты использования стратегий A
        'y_freq': y_freq.copy(),  # Частоты использования стратегий B
        'upper': upper,
        'lower': lower,
        'epsilon': epsilon
    }
    
    if k <= 10:
        first_10.append(record)
    last_10.append(record)
    
    if epsilon <= 0.1:
        break

# Формирование таблицы (частоты использования стратегий, как в документе)
table_data = []
for record in first_10 + list(last_10):
    row = [
        record['k'],
        record['choice_A'],
        record['choice_B'],
        *record['x_freq'],  # Частоты использования стратегий A
        *record['y_freq'],  # Частоты использования стратегий B
        f"{record['upper']:.4f}",
        f"{record['lower']:.4f}", 
        f"{record['epsilon']:.6f}"
    ]
    table_data.append(row)

headers = ['k', 'A', 'B', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'Верхняя', 'Нижняя', 'ε']

print("Результаты метода Брауна-Робинсона (первые 10 и последние 10 итераций):")
print(tabulate(table_data, headers=headers, tablefmt='grid'))

# Финальные результаты (средние стратегии)
x_avg = x_freq / k
y_avg = y_freq / k
v_approx = (upper + lower) / 2

print(f"\nФинальные результаты после {k} итераций:")
print(f"Приближенная цена игры: {v_approx:.6f}")
print(f"Погрешность: {epsilon:.6f}")
print(f"Стратегия игрока A: [{', '.join([f'{x:.6f}' for x in x_avg])}]")
print(f"Стратегия игрока B: [{', '.join([f'{y:.6f}' for y in y_avg])}]")

# Сравнение с аналитическим методом
print("\n" + "=" * 50)
print("СРАВНЕНИЕ С АНАЛИТИЧЕСКИМ МЕТОДОМ")
print("=" * 50)
print(f"Аналитическая цена игры: {v_analytic:.6f}")
print(f"Численная цена игры: {v_approx:.6f}")
print(f"Разница: {abs(v_analytic - v_approx):.6f}")

# Проверка: должны получить стратегии, близкие к аналитическим
print(f"\nАналитическая стратегия A: {x_analytic}")
print(f"Численная стратегия A: {x_avg}")
print(f"Аналитическая стратегия B: {y_analytic}")
print(f"Численная стратегия B: {y_avg}")