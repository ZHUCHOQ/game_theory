import random
import numpy as np

# Параметры генерации случайной игры
GAME_SIZE = (10, 10)
MIN_VAL = -100
MAX_VAL = 100

# ANSI коды для цветового выделения
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def generate_bimatrix(rows, cols, min_val, max_val):
    """Генерация случайной биматричной игры"""
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            p1 = random.randint(min_val, max_val)
            p2 = random.randint(min_val, max_val)
            row.append((p1, p2))
        matrix.append(row)
    return matrix

def find_pareto_optimal(matrix):
    """Поиск Парето-оптимальных ситуаций"""
    pareto_optimal = []
    rows = len(matrix)
    cols = len(matrix[0])
    
    for i in range(rows):
        for j in range(cols):
            current = matrix[i][j]
            is_dominated = False
            
            for x in range(rows):
                for y in range(cols):
                    other = matrix[x][y]
                    if (other[0] >= current[0] and other[1] >= current[1]) and \
                       (other[0] > current[0] or other[1] > current[1]):
                        is_dominated = True
                        break
                if is_dominated:
                    break
            
            if not is_dominated:
                pareto_optimal.append((i, j))
                
    return pareto_optimal

def find_nash_equilibria(matrix):
    """Поиск равновесий Нэша в чистых стратегиях"""
    nash_equilibria = []
    rows = len(matrix)
    cols = len(matrix[0])
    
    for i in range(rows):
        for j in range(cols):
            current = matrix[i][j]
            
            # Проверка оптимальности для первого игрока
            is_p1_optimal = True
            for x in range(rows):
                if matrix[x][j][0] > current[0]:
                    is_p1_optimal = False
                    break
            
            # Проверка оптимальности для второго игрока
            is_p2_optimal = True
            for y in range(cols):
                if matrix[i][y][1] > current[1]:
                    is_p2_optimal = False
                    break
            
            if is_p1_optimal and is_p2_optimal:
                nash_equilibria.append((i, j))
                
    return nash_equilibria

def analyze_2x2_game(matrix):
    """Анализ биматричной игры 2×2"""
    # Поиск равновесий в чистых стратегиях
    pure_nash = find_nash_equilibria(matrix)
    
    # Проверка наличия смешанного равновесия
    A = np.array([[matrix[0][0][0], matrix[0][1][0]],
                 [matrix[1][0][0], matrix[1][1][0]]])
    B = np.array([[matrix[0][0][1], matrix[0][1][1]],
                 [matrix[1][0][1], matrix[1][1][1]]])
    
    mixed_nash = None
    payoffs = None
    
    try:
        u = np.array([1, 1])
        A_inv = np.linalg.inv(A)
        B_inv = np.linalg.inv(B)
        
        v1 = 1 / (u @ A_inv @ u)
        v2 = 1 / (u @ B_inv @ u)
        
        x = v2 * (u @ B_inv)
        y = v1 * (A_inv @ u)
        
        # Проверка, что вероятности в допустимом диапазоне
        if all(0 <= p <= 1 for p in x) and all(0 <= p <= 1 for p in y):
            mixed_nash = (x.tolist(), y.tolist())
            payoffs = (v1, v2)
    except np.linalg.LinAlgError:
        # Матрицы вырождены, смешанного равновесия нет
        pass
    
    return pure_nash, mixed_nash, payoffs

def print_matrix_with_highlight(matrix, pareto_indices=None, nash_indices=None):
    """Вывод матрицы с выделением оптимальных стратегий"""
    rows = len(matrix)
    cols = len(matrix[0])
    
    print("Матрица выигрышей с выделением оптимальных стратегий:")
    for i in range(rows):
        for j in range(cols):
            cell = f"({matrix[i][j][0]}, {matrix[i][j][1]})"
            
            # Проверка, нужно ли выделять ячейку
            is_pareto = pareto_indices and (i, j) in pareto_indices
            is_nash = nash_indices and (i, j) in nash_indices
            
            if is_pareto and is_nash:
                cell = f"{Colors.BLUE}{Colors.BOLD}{Colors.UNDERLINE}{cell}{Colors.RESET}"
            elif is_pareto:
                cell = f"{Colors.RED}{Colors.BOLD}{cell}{Colors.RESET}"
            elif is_nash:
                cell = f"{Colors.GREEN}{Colors.UNDERLINE}{cell}{Colors.RESET}"
                
            print(cell, end=" ")
        print()

def print_detailed_analysis(matrix, pareto_indices, nash_indices, game_name=""):
    """Подробный анализ оптимальных ситуаций"""
    if game_name:
        print(f"\nАнализ игры '{game_name}':")
    
    print("Парето-оптимальные ситуации:")
    for i, j in pareto_indices:
        payoff = matrix[i][j]
        print(f"  Позиция ({i}, {j}): {payoff}")
        
    print("\nРавновесия Нэша:")
    for i, j in nash_indices:
        payoff = matrix[i][j]
        print(f"  Позиция ({i}, {j}): {payoff}")
    
    # Пересечение множеств (ситуации, оптимальные по обоим критериям)
    intersection = set(pareto_indices) & set(nash_indices)
    print("\nПересечение множеств (Парето-оптимальные и равновесия Нэша):")
    if intersection:
        for i, j in intersection:
            payoff = matrix[i][j]
            print(f"  Позиция ({i}, {j}): {payoff}")
    else:
        print("  Пересечений нет")

def analyze_known_game(matrix, game_name):
    """Анализ известной игры"""
    print(f"\n{'-'*60}")
    print(f"Анализ игры: {game_name}")
    print(f"{'-'*60}")
    
    # Поиск оптимальных ситуаций
    pareto_optimal = find_pareto_optimal(matrix)
    nash_equilibria = find_nash_equilibria(matrix)
    
    # Вывод матрицы с выделением оптимальных стратегий
    print_matrix_with_highlight(matrix, pareto_optimal, nash_equilibria)
    
    # Подробный анализ
    print_detailed_analysis(matrix, pareto_optimal, nash_equilibria, game_name)

# Классическая дилемма заключенного
def classical_prisoners_dilemma():
    """Классическая дилемма заключенного"""
    return [
        [(-1, -1), (-3, 0)],
        [(0, -3), (-2, -2)]
    ]

# Основная часть программы
if __name__ == "__main__":
    # Задание 1: Случайная игра 10×10
    random.seed(42)
    game_matrix = generate_bimatrix(10, 10, MIN_VAL, MAX_VAL)
    
    print("=" * 60)
    print("ЗАДАНИЕ 1: Случайная биматричная игра 10×10")
    print("=" * 60)
    
    # Поиск оптимальных ситуаций
    pareto_optimal = find_pareto_optimal(game_matrix)
    nash_equilibria = find_nash_equilibria(game_matrix)
    
    # Вывод результатов
    print("Парето-оптимальные ситуации (индексы строк и столбцов):", pareto_optimal)
    print("Равновесия Нэша в чистых стратегиях:", nash_equilibria)
    
    # Вывод матрицы с выделением оптимальных стратегий
    print_matrix_with_highlight(game_matrix, pareto_optimal, nash_equilibria)
    
    # Подробный анализ оптимальных ситуаций
    print_detailed_analysis(game_matrix, pareto_optimal, nash_equilibria, "Случайная игра 10×10")
    
    # Задание 2: Анализ игры для варианта 9
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2: Анализ игры для варианта 9")
    print("=" * 60)
    
    variant_9_matrix = [
        [(5, 1), (10, 4)],
        [(8, 6), (6, 9)]
    ]
    
    # Анализ игры
    pure_nash, mixed_nash, payoffs = analyze_2x2_game(variant_9_matrix)
    
    # Вывод матрицы игры
    print("Матрица выигрышей для варианта 9:")
    print_matrix_with_highlight(variant_9_matrix)
    
    # Вывод результатов анализа
    print("\nРезультаты анализа:")
    print("Равновесия Нэша в чистых стратегиях:", pure_nash)
    
    if mixed_nash:
        print("Смешанное равновесие:")
        print(f"  Стратегия первого игрока: {mixed_nash[0]}")
        print(f"  Стратегия второго игрока: {mixed_nash[1]}")
        print(f"  Ожидаемые выигрыши: {payoffs}")
    else:
        print("Смешанного равновесия не найдено")
    
    # Дополнительный анализ для варианта 9
    print("\nДополнительный анализ:")
    if pure_nash:
        for i, j in pure_nash:
            print(f"В ситуации ({i}, {j}):")
            print(f"  Выигрыш первого игрока: {variant_9_matrix[i][j][0]}")
            print(f"  Выигрыш второго игрока: {variant_9_matrix[i][j][1]}")
            
            # Проверка устойчивости
            print("  Проверка устойчивости:")
            # Для первого игрока
            other_payoff = variant_9_matrix[1-i][j][0] if i == 0 else variant_9_matrix[0][j][0]
            if variant_9_matrix[i][j][0] >= other_payoff:
                print(f"    Первому игроку невыгодно менять стратегию ({variant_9_matrix[i][j][0]} >= {other_payoff})")
            else:
                print(f"    Первому игроку выгодно менять стратегию ({variant_9_matrix[i][j][0]} < {other_payoff})")
            
            # Для второго игрока
            other_payoff = variant_9_matrix[i][1-j][1] if j == 0 else variant_9_matrix[i][0][1]
            if variant_9_matrix[i][j][1] >= other_payoff:
                print(f"    Второму игроку невыгодно менять стратегию ({variant_9_matrix[i][j][1]} >= {other_payoff})")
            else:
                print(f"    Второму игроку выгодно менять стратегию ({variant_9_matrix[i][j][1]} < {other_payoff})")
    
    # Анализ известных игр
    print("\n" + "=" * 60)
    print("АНАЛИЗ ИЗВЕСТНЫХ ИГР")
    print("=" * 60)
    
    # 1. Семейный спор (Battle of the Sexes)
    battle_of_sexes = [
        [(4, 1), (0, 0)],
        [(0, 0), (1, 4)]
    ]
    analyze_known_game(battle_of_sexes, "Семейный спор")
    
    # 2. Перекресток (Crossroad)
    # Для этой игры добавим небольшой ε, чтобы избежать вырожденности
    epsilon = 0.001
    crossroad = [
        [(1.0, 1.0), (1-epsilon, 2.0)],
        [(2.0, 1-epsilon), (0.0, 0.0)]
    ]
    analyze_known_game(crossroad, "Перекресток")
    
    # 3. Дилемма заключенного (Prisoner's Dilemma)
    prisoners_dilemma = [
        [(-5, -5), (0, -10)],
        [(-10, 0), (-1, -1)]
    ]
    analyze_known_game(prisoners_dilemma, "Дилемма заключенного (заданная)")
    
    # 4. Классическая дилемма заключенного
    classical_pd = classical_prisoners_dilemma()
    analyze_known_game(classical_pd, "Классическая дилемма заключенного")