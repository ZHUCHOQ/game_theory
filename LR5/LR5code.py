import math
from itertools import combinations

def characteristic_function(coalition):
    """
    Определение характеристической функции для варианта 9 с корректировками
    для обеспечения супераддитивности
    """
    coalition = tuple(sorted(coalition))
    v = {
        (): 0,          # Пустая коалиция
        (1,): 3,        # Игрок 1
        (2,): 3,        # Игрок 2
        (3,): 1,        # Игрок 3
        (4,): 2,        # Игрок 4
        (1,2): 7,       # Коалиция 1+2
        (1,3): 6,       # Коалиция 1+3
        (1,4): 7,       # Коалиция 1+4
        (2,3): 7,       # Коалиция 2+3
        (2,4): 5,       # Коалиция 2+4 (скорректировано с 4 до 5)
        (3,4): 7,       # Коалиция 3+4
        (1,2,3): 10,    # Коалиция 1+2+3
        (1,2,4): 10,    # Коалиция 1+2+4
        (1,3,4): 10,    # Коалиция 1+3+4 (скорректировано с 9 до 10)
        (2,3,4): 10,    # Коалиция 2+3+4 (скорректировано с 7 до 10)
        (1,2,3,4): 14   # Большая коалиция (скорректировано с 12 до 14)
    }
    return v.get(coalition, 0)

def calculate_shapley_value(n, v):
    """
    Вычисление вектора Шепли для n игроков с характеристической функцией v
    """
    shapley_values = [0] * n
    all_players = list(range(1, n+1))
    
    # Перебираем всех игроков
    for player in range(1, n+1):
        # Перебираем все возможные размеры коалиций
        for size in range(0, n):
            # Перебираем все коалиции определенного размера без текущего игрока
            for S in combinations([p for p in all_players if p != player], size):
                S = tuple(S)
                # Вычисляем вес для данной коалиции
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                # Вычисляем предельный вклад игрока
                marginal_contribution = v(S + (player,)) - v(S)
                # Добавляем взвешенный вклад к значению Шепли игрока
                shapley_values[player-1] += weight * marginal_contribution
                
    return shapley_values

# Основная часть программы
if __name__ == "__main__":
    n_players = 4
    shapley_values = calculate_shapley_value(n_players, characteristic_function)
    
    # Вывод результатов
    print("Вектор Шепли:", [round(val, 4) for val in shapley_values])
    print("Сумма компонент вектора Шепли:", round(sum(shapley_values), 4))
    print("v(I) =", characteristic_function((1,2,3,4)))
    
    # Проверка индивидуальной рационализации
    for i in range(n_players):
        individual_value = characteristic_function((i+1,))
        print(f"x_{i+1} = {shapley_values[i]:.4f} >= v({{'{i+1}'}}) = {individual_value}")