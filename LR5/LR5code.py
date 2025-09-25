import math
from itertools import combinations

def check_superadditivity(v, n_players):
    """
    Проверка супераддитивности и автоматическая корректировка характеристической функции
    """
    print("=== ПРОВЕРКА СУПЕРАДДИТИВНОСТИ ===")
    
    # Исходные данные для варианта 9
    original_v = {
        (): 0, (1,): 3, (2,): 3, (3,): 1, (4,): 2,
        (1,2): 7, (1,3): 6, (1,4): 7, (2,3): 7, (2,4): 4, (3,4): 7,
        (1,2,3): 10, (1,2,4): 10, (1,3,4): 9, (2,3,4): 7, (1,2,3,4): 12
    }
    
    # Начинаем с исходных значений
    corrected_v = original_v.copy()
    violations = []
    
    # Проверка всех возможных разбиений коалиций
    all_players = tuple(range(1, n_players+1))
    all_coalitions = []
    
    # Генерируем все возможные коалиции
    for size in range(1, n_players+1):
        for coalition in combinations(all_players, size):
            all_coalitions.append(tuple(sorted(coalition)))
    
    # Проверяем супераддитивность для всех пар непересекающихся коалиций
    for coalition in all_coalitions:
        if len(coalition) == 0:
            continue
            
        # Генерируем все возможные разбиения коалиции на две непустые части
        for split_point in range(1, len(coalition)):
            for part1 in combinations(coalition, split_point):
                part1 = tuple(sorted(part1))
                part2 = tuple(sorted(set(coalition) - set(part1)))
                
                if len(part2) == 0:
                    continue
                
                current_value = corrected_v.get(coalition, 0)
                sum_parts = corrected_v.get(part1, 0) + corrected_v.get(part2, 0)
                
                if current_value < sum_parts:
                    violations.append((coalition, part1, part2, current_value, sum_parts))
                    # Корректируем значение
                    corrected_v[coalition] = sum_parts
                    print(f"НАРУШЕНИЕ: v{coalition} = {current_value} < v{part1} + v{part2} = {sum_parts}")
                    print(f"КОРРЕКТИРОВКА: v{coalition} = {sum_parts}")
    
    print(f"\nОбнаружено нарушений: {len(violations)}")
    return corrected_v, violations

def characteristic_function(coalition, corrected_values):
    """
    Характеристическая функция с использованием скорректированных значений
    """
    coalition = tuple(sorted(coalition))
    return corrected_values.get(coalition, 0)

def calculate_shapley_value(n, v):
    """
    Вычисление вектора Шепли
    """
    shapley_values = [0] * n
    all_players = list(range(1, n+1))
    
    for player in range(1, n+1):
        for size in range(0, n):
            for S in combinations([p for p in all_players if p != player], size):
                S = tuple(S)
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                marginal_contribution = v(S + (player,)) - v(S)
                shapley_values[player-1] += weight * marginal_contribution
                
    return shapley_values

# Основная программа
if __name__ == "__main__":
    n_players = 4
    
    # Шаг 1: Проверка и корректировка характеристической функции
    corrected_v, violations = check_superadditivity(None, n_players)
    
    print("\n=== СКОРРЕКТИРОВАННАЯ ХАРАКТЕРИСТИЧЕСКАЯ ФУНКЦИЯ ===")
    for coalition in sorted(corrected_v.keys(), key=lambda x: (len(x), x)):
        print(f"v{coalition} = {corrected_v[coalition]}")
    
    # Шаг 2: Вычисление вектора Шепли
    shapley_values = calculate_shapley_value(n_players, 
                                           lambda coalition: characteristic_function(coalition, corrected_v))
    
    print("\n=== РЕЗУЛЬТАТЫ РАСЧЕТА ===")
    print("Вектор Шепли:", [round(val, 4) for val in shapley_values])
    print("Сумма компонент вектора Шепли:", round(sum(shapley_values), 4))
    print("v(I) =", corrected_v[tuple(range(1, n_players+1))])
    
    # Шаг 3: Проверка рационализации
    print("\n=== ПРОВЕРКА РАЦИОНАЛИЗАЦИИ ===")
    # Групповая рационализация
    group_sum = sum(shapley_values)
    v_I = corrected_v[tuple(range(1, n_players+1))]
    print(f"Групповая рационализация: {group_sum:.4f} = {v_I} ✓" 
          if abs(group_sum - v_I) < 0.001 
          else f"Групповая рационализация: {group_sum:.4f} ≠ {v_I} ✗")
    
    # Индивидуальная рационализация
    for i in range(n_players):
        individual_value = corrected_v.get((i+1,), 0)
        shapley_val = shapley_values[i]
        status = "✓" if shapley_val >= individual_value else "✗"
        print(f"x_{i+1} = {shapley_val:.4f} >= v({{'{i+1}'}}) = {individual_value} {status}")