import random
import numpy as np

# Параметры генерации случайной игры
GAME_SIZE = (10, 10)
MIN_VAL = -100
MAX_VAL = 100

def generate_bimatrix(rows, cols, min_val, max_val):
    """Генерация случайной биматричной игры"""
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # Случайные выигрыши для двух игроков
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
            
            # Проверка, доминируется ли текущая ситуация другими
            for x in range(rows):
                for y in range(cols):
                    other = matrix[x][y]
                    # Ситуация доминируется, если существует другая ситуация,
                    # которая не хуже по всем показателям и лучше хотя бы по одному
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
    
    # Вычисление смешанного равновесия по формулам
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
        else:
            mixed_nash = None
            payoffs = None
    except:
        mixed_nash = None
        payoffs = None
    
    return pure_nash, mixed_nash, payoffs

def print_matrix_with_highlight(matrix, pareto_indices=None, nash_indices=None):
    """Вывод матрицы с выделением оптимальных стратегий"""
    rows = len(matrix)
    cols = len(matrix[0])
    
    # ANSI коды для цветового выделения
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    print("Матрица выигрышей:")
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            cell = f"({matrix[i][j][0]}, {matrix[i][j][1]})"
            
            # Проверка, нужно ли выделять ячейку
            is_pareto = pareto_indices and (i, j) in pareto_indices
            is_nash = nash_indices and (i, j) in nash_indices
            
            if is_pareto and is_nash:
                cell = f"{BLUE}{BOLD}{UNDERLINE}{cell}{RESET}"
            elif is_pareto:
                cell = f"{RED}{BOLD}{cell}{RESET}"
            elif is_nash:
                cell = f"{GREEN}{UNDERLINE}{cell}{RESET}"
                
            row_str += cell + " "
        print(row_str)

# Основная часть программы
if __name__ == "__main__":
    # Задание 1: Случайная игра 10×10
    random.seed(42)  # Для воспроизводимости результатов
    game_matrix = generate_bimatrix(10, 10, MIN_VAL, MAX_VAL)
    
    print("=" * 50)
    print("ЗАДАНИЕ 1: Случайная биматричная игра 10×10")
    print("=" * 50)
    
    # Поиск оптимальных ситуаций
    pareto_optimal = find_pareto_optimal(game_matrix)
    nash_equilibria = find_nash_equilibria(game_matrix)
    
    # Вывод результатов
    print("Парето-оптимальные ситуации (индексы строк и столбцов):", pareto_optimal)
    print("Равновесия Нэша в чистых стратегиях:", nash_equilibria)
    
    # Вывод матрицы с выделением оптимальных стратегий
    print_matrix_with_highlight(game_matrix, pareto_optimal, nash_equilibria)
    
    # Задание 2: Анализ игры 2×2 для варианта 9
    print("\n" + "=" * 50)
    print("ЗАДАНИЕ 2: Анализ игры для варианта 9")
    print("=" * 50)
    
    variant_9_matrix = [
        [(5, 1), (10, 4)],
        [(8, 6), (6, 9)]
    ]
    
    # Анализ игры
    pure_nash, mixed_nash, payoffs = analyze_2x2_game(variant_9_matrix)
    
    # Вывод матрицы игры
    print("Матрица выигрышей для варианта 9:")
    print("Первый игрок (строка):")
    print(f"  Стратегия 1: {variant_9_matrix[0][0]}, {variant_9_matrix[0][1]}")
    print(f"  Стратегия 2: {variant_9_matrix[1][0]}, {variant_9_matrix[1][1]}")
    print("Второй игрок (столбец):")
    print(f"  Стратегия 1: {variant_9_matrix[0][0]}, {variant_9_matrix[1][0]}")
    print(f"  Стратегия 2: {variant_9_matrix[0][1]}, {variant_9_matrix[1][1]}")
    
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