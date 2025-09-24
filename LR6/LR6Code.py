import numpy as np


def simulate_opinion_dynamics(n_agents=9, epsilon=1e-6, max_iter=1000):
    """
    Моделирование динамики мнений в социальной сети с подробным выводом информации
    """
    
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №8: МОДЕЛИРОВАНИЕ ИНФОРМАЦИОННОГО ПРОТИВОБОРСТВА")
    print("=" * 60)
    
    # 1. Генерация стохастической матрицы доверия
    print("\n1. ГЕНЕРАЦИЯ СТОХАСТИЧЕСКОЙ МАТРИЦЫ ДОВЕРИЯ")
    print("-" * 50)
    
    np.random.seed(42)  # Для воспроизводимости результатов
    A = np.random.rand(n_agents, n_agents)
    A = A / A.sum(axis=1, keepdims=True)
    np.set_printoptions(precision=3, suppress=True)
    
    print("Сгенерированная матрица доверия:")
    for i in range(n_agents):
        print(f"Строка {i}: {A[i]}")
    
    row_sums = A.sum(axis=1)
    print("\nСуммы строк матрицы (должны быть равны 1):")
    for i, sum_val in enumerate(row_sums):
        print(f"Строка {i}: {sum_val:.6f}")
    
    # 2. Моделирование без управления
    print("\n\n2. МОДЕЛИРОВАНИЕ БЕЗ УПРАВЛЕНИЯ")
    print("-" * 50)
    
    x0 = np.random.uniform(1, 20, n_agents)
    print("Начальные мнения агентов:")
    for i, opinion in enumerate(x0):
        print(f"Агент {i}: {opinion:.3f}")
    
    # Итерационный процесс
    x = x0.copy()
    history = [x.copy()]
    
    print("\nПроцесс итераций (первые 5 итераций):")
    print(f"Итерация 0: {x}")
    
    for i in range(max_iter):
        x_new = A @ x
        history.append(x_new.copy())
        
        if i < 5:
            print(f"Итерация {i+1}: {x_new}")
        
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"...\nСходимость достигнута на итерации {i+1}")
            break
        x = x_new
    
    print("\nИтоговые мнения агентов:")
    for i, opinion in enumerate(x):
        print(f"Агент {i}: {opinion:.3f}")
    

    # 3. Моделирование с информационным управлением
    print("\n\n3. МОДЕЛИРОВАНИЕ С ИНФОРМАЦИОННЫМ УПРАВЛЕНИЕМ")
    print("-" * 50)
    
    # Выбор агентов влияния
    agents = np.arange(n_agents)
    np.random.shuffle(agents)
    player1_agents = agents[:2]
    player2_agents = agents[2:4]
    neutral_agents = agents[4:]
    
    print("Агенты первого игрока:", player1_agents)
    print("Агенты второго игрока:", player2_agents)
    print("Нейтральные агенты:", neutral_agents)
    
    # Формирование начальных мнений
    x0_controlled = np.random.uniform(1, 20, n_agents)
    x0_controlled[player1_agents] = np.random.uniform(0, 100, len(player1_agents))
    x0_controlled[player2_agents] = np.random.uniform(-100, 0, len(player2_agents))
    
    print("\nНачальные мнения с учетом управления:")
    for i, opinion in enumerate(x0_controlled):
        agent_type = ""
        if i in player1_agents:
            agent_type = " (игрок 1)"
        elif i in player2_agents:
            agent_type = " (игрок 2)"
        else:
            agent_type = " (нейтральный)"
        print(f"Агент {i}{agent_type}: {opinion:.3f}")
    
    # Итерационный процесс
    x = x0_controlled.copy()
    history_controlled = [x.copy()]
    
    print("\nПроцесс итераций (первые 5 итераций):")
    print(f"Итерация 0: {x}")
    
    for i in range(max_iter):
        x_new = A @ x
        history_controlled.append(x_new.copy())
        
        if i < 5:
            print(f"Итерация {i+1}: {x_new}")
        
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"...\nСходимость достигнута на итерации {i+1}")
            break
        x = x_new
    
    print("\nИтоговые мнения агентов:")
    for i, opinion in enumerate(x):
        print(f"Агент {i}: {opinion:.3f}")
    
    # Определение победителя
    final_opinion = x[0]
    print(f"\nИТОГОВОЕ МНЕНИЕ: {final_opinion:.3f}")
    
    if final_opinion > 0:
        print("ВЫВОД: Выиграл первый игрок (итоговое мнение положительное)")
    else:
        print("ВЫВОД: Выиграл второй игрок (итоговое мнение отрицательное)")
    
    # 4. Анализ матрицы влияния
    print("\n\n4. АНАЛИЗ МАТРИЦЫ ВЛИЯНИЯ")
    print("-" * 50)
    
    # Вычисление стационарного распределения
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary_distribution = np.real(eigenvectors[:, stationary_idx])
    stationary_distribution = np.abs(stationary_distribution)  # Берем модуль
    stationary_distribution /= stationary_distribution.sum()  # Нормализуем
    
    # Предельная матрица влияния
    A_infinity = np.tile(stationary_distribution, (n_agents, 1))
    
    print("Предельная матрица влияния (первые 3 строки):")
    for i in range(min(3, n_agents)):
        print(f"Строка {i}: {A_infinity[i]}")
    
    print("\nВлияние агентов (стационарное распределение):")
    for i, score in enumerate(stationary_distribution):
        print(f"Агент {i}: {score:.4f}")
    
    # Рейтинг агентов по влиянию
    most_influential = np.argsort(stationary_distribution)[::-1]
    print("\nРейтинг агентов по влиянию:")
    for rank, agent in enumerate(most_influential):
        print(f"{rank+1}. Агент {agent} (влияние: {stationary_distribution[agent]:.4f})")
    
    # 5. Дополнительный анализ
    print("\n\n5. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("-" * 50)
    
    final_no_control = history[-1][0]
    final_with_control = history_controlled[-1][0]
    
    print(f"Итоговое мнение без управления: {final_no_control:.3f}")
    print(f"Итоговое мнение с управлением: {final_with_control:.3f}")
    
    control_effect = final_with_control - final_no_control
    print(f"Изменение мнения под воздействием управления: {control_effect:.3f}")
    
    if control_effect > 0:
        print("Управление сместило мнение в положительную сторону")
    else:
        print("Управление сместило мнение в отрицательную сторону")
    
    # Анализ вклада агентов
    print("\nАнализ вклада агентов в итоговое мнение:")
    weighted_sum = 0
    for i, (weight, opinion) in enumerate(zip(stationary_distribution, x0_controlled)):
        contribution = weight * opinion
        weighted_sum += contribution
        
        agent_type = ""
        if i in player1_agents:
            agent_type = " (игрок 1)"
        elif i in player2_agents:
            agent_type = " (игрок 2)"
        else:
            agent_type = " (нейтральный)"
        
        print(f"Агент {i}{agent_type}: вес={weight:.4f}, мнение={opinion:.3f}, вклад={contribution:.3f}")
    
    print(f"\nСумма вкладов: {weighted_sum:.3f}")
    print(f"Итоговое мнение: {final_with_control:.3f}")
    print(f"Разница: {abs(weighted_sum - final_with_control):.6f} (должна быть близка к 0)")
    
    return A, x0, x0_controlled, stationary_distribution

# Запуск моделирования
if __name__ == "__main__":
    simulate_opinion_dynamics()