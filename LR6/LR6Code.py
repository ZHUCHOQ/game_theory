import numpy as np

def simulate_opinion_dynamics(n_agents=9, epsilon=1e-6, max_iter=1000):
    """
    Моделирование динамики мнений в социальной сети
    
    Parameters:
    n_agents: количество агентов
    epsilon: точность схождения
    max_iter: максимальное количество итераций
    """
    
    # 1. Генерация стохастической матрицы доверия
    print("1. Стохастическая матрица доверия:")
    A = np.random.rand(n_agents, n_agents)
    A = A / A.sum(axis=1, keepdims=True)  # Нормализация строк
    np.set_printoptions(precision=3, suppress=True)
    print(A)
    
    # 2. Моделирование без управления
    print("\n2. Моделирование без управления:")
    x0 = np.random.uniform(1, 20, n_agents)
    print("Начальные мнения агентов:", x0)
    
    # Итерационный процесс
    x = x0.copy()
    for i in range(max_iter):
        x_new = A @ x
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"Сходимость достигнута на итерации {i+1}")
            break
        x = x_new
    else:
        print("Достигнуто максимальное количество итераций")
    
    print("Итоговые мнения агентов:", x)
    
    # 3. Моделирование с информационным управлением
    print("\n3. Моделирование с информационным управлением:")
    
    # Выбор агентов влияния
    agents = np.arange(n_agents)
    np.random.shuffle(agents)
    player1_agents = agents[:2]
    player2_agents = agents[2:4]
    neutral_agents = agents[4:]
    
    print("Агенты первого игрока:", player1_agents)
    print("Агенты второго игрока:", player2_agents)
    print("Нейтральные агенты:", neutral_agents)
    
    # Формирование начальных мнений с учетом управления
    x0_controlled = np.random.uniform(1, 20, n_agents)
    x0_controlled[player1_agents] = np.random.uniform(0, 100, len(player1_agents))
    x0_controlled[player2_agents] = np.random.uniform(-100, 0, len(player2_agents))
    
    print("Начальные мнения с учетом управления:", x0_controlled)
    
    # Итерационный процесс
    x = x0_controlled.copy()
    for i in range(max_iter):
        x_new = A @ x
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"Сходимость достигнута на итерации {i+1}")
            break
        x = x_new
    else:
        print("Достигнуто максимальное количество итераций")
    
    print("Итоговые мнения агентов:", x)
    
    # Определение победителя
    final_opinion = x[0]  # Все мнения одинаковы
    if final_opinion > 0:
        print("Выиграл первый игрок")
    else:
        print("Выиграл второй игрок")
    
    return A, x0, x0_controlled, x

# Запуск моделирования
if __name__ == "__main__":
    simulate_opinion_dynamics()