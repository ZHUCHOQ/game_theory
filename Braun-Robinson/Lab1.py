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

# Инициализация
n = 3
x_count = np.zeros(n)  # Счетчик стратегий игрока A
y_count = np.zeros(n)  # Счетчик стратегий игрока B

# Накопленные суммы выигрышей для каждой стратегии
A_accumulated = np.zeros(n)  # Для игрока A
B_accumulated = np.zeros(n)  # Для игрока B

# История для таблицы (первые 10 и последние 10 итераций)
first_10 = []
last_10 = deque(maxlen=10)  # deque для хранения последних 10 итераций

# История для графиков (каждые 50 итераций)
history_for_plots = []

# Максимальное число итераций
max_iterations = 10000

# Итерационный процесс
for k in range(1, max_iterations + 1):
    # Выбор стратегий на текущей итерации
    if k == 1:
        # На первой итерации выбираем произвольно
        choice_A = 0
        choice_B = 0
    else:
        # Игрок A выбирает стратегию с максимальным накопленным выигрышем
        choice_A = np.argmax(A_accumulated)
        # Игрок B выбирает стратегию с минимальным накопленным проигрышем
        choice_B = np.argmin(B_accumulated)
    
    # Обновляем счетчики стратегий
    x_count[choice_A] += 1
    y_count[choice_B] += 1
    
    # Обновляем накопленные суммы выигрышей
    for i in range(n):
        A_accumulated[i] += C[i, choice_B]
    
    for j in range(n):
        B_accumulated[j] += C[choice_A, j]
    
    # Вычисляем оценки цены игры
    upper = np.max(A_accumulated) / k
    lower = np.min(B_accumulated) / k
    epsilon = upper - lower
    
    # Формируем запись для истории
    record = {
        'k': k,
        'choice_A': f'x{choice_A + 1}',
        'choice_B': f'y{choice_B + 1}',
        'A_accumulated': A_accumulated.copy(),
        'B_accumulated': B_accumulated.copy(),
        'upper': upper,
        'lower': lower,
        'epsilon': epsilon
    }
    
    # Сохраняем первые 10 итераций
    if k <= 10:
        first_10.append(record)
    
    # Сохраняем последние 10 итераций
    last_10.append(record)
    
    # Сохраняем для графиков каждые 50 итераций
    if k % 50 == 0:
        history_for_plots.append(record)
    
    # Проверка условия останова
    if epsilon <= 0.1:
        break

# Формируем итоговую таблицу из первых 10 и последних 10 итераций
table_data = []
for record in first_10 + list(last_10):
    row = [
        record['k'],
        record['choice_A'],
        record['choice_B'],
        *record['A_accumulated'],
        *record['B_accumulated'],
        f"{record['upper']:.2f}",
        f"{record['lower']:.2f}",
        f"{record['epsilon']:.4f}"
    ]
    table_data.append(row)

# Заголовки таблицы
headers = [
    'k', 'A', 'B',
    'x1', 'x2', 'x3',
    'y1', 'y2', 'y3',
    'Верхняя', 'Нижняя', 'ε'
]

# Выводим таблицу
html_table = tabulate(table_data, headers=headers, tablefmt='html', floatfmt=".4f")

# Сохраняем таблицу в текстовый файл для удобного копирования
with open('results_table.html', 'w', encoding='utf-8') as f:
    f.write(html_table)

# Финальные результаты
x_avg = x_count / k
y_avg = y_count / k
upper = np.max(A_accumulated) / k
lower = np.min(B_accumulated) / k
epsilon = upper - lower
v_approx = (upper + lower) / 2

print(f"\nФинальные результаты после {k} итераций:")
print(f"Приближенная цена игры: {v_approx:.3f}")
print(f"Погрешность: {epsilon:.4f}")
print(f"Стратегия игрока A: {x_avg}")
print(f"Стратегия игрока B: {y_avg}")