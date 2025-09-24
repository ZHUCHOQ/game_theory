import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# 1. ГЕНЕРАЦИЯ СТОХАСТИЧЕСКОЙ МАТРИЦЫ ДОВЕРИЯ ДЛЯ 10 АГЕНТОВ
# =============================================================================

def generate_stochastic_matrix(n, seed=42):
    """Генерирует стохастическую матрицу с более равномерным распределением"""
    np.random.seed(seed)
    matrix = np.random.rand(n, n) + 0.1  # Добавляем 0.1 чтобы избежать нулей
    stochastic_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return stochastic_matrix

# Генерируем матрицу доверия для 10 агентов
n_agents = 10
A = generate_stochastic_matrix(n_agents)

print("1. СТОХАСТИЧЕСКАЯ МАТРИЦА ДОВЕРИЯ A:")
print("Размер матрицы:", A.shape)
print("Проверка: сумма каждой строки должна быть равна 1")
print("Суммы строк:", np.sum(A, axis=1).round(3))
print("\nМатрица A:")
print(A.round(3))

# =============================================================================
# 2. ВЫЧИСЛЕНИЕ РЕЗУЛЬТИРУЮЩЕЙ МАТРИЦЫ A∞ (ПРЕДЕЛЬНОЕ РАСПРЕДЕЛЕНИЕ)
# =============================================================================

def compute_limiting_matrix(A, tolerance=1e-10, max_iterations=1000):
    """Вычисляет предельную матрицу A∞ методом итерационного возведения в степень"""
    A_current = A.copy()
    
    for i in range(max_iterations):
        A_next = np.dot(A_current, A)
        max_diff = np.max(np.abs(A_next - A_current))
        
        if max_diff < tolerance:
            print(f"Сходимость достигнута на итерации {i+1}")
            break
            
        A_current = A_next
    
    return A_current

# Вычисляем A∞
A_inf = compute_limiting_matrix(A)

print("\n" + "="*80)
print("2. РЕЗУЛЬТИРУЮЩАЯ МАТРИЦА A∞:")
print("Все строки должны быть одинаковыми (стационарное распределение)")
print("\nМатрица A∞ (первые 3 строки):")
print(A_inf[:3].round(3))

# Вектор r - первая строка матрицы A∞ (все строки одинаковы)
r = A_inf[0, :]
print(f"\nВектор r (стационарное распределение): {r.round(3)}")
print(f"Сумма элементов r: {r.sum():.6f}")

# =============================================================================
# 3. ВЫБОР АГЕНТОВ ВЛИЯНИЯ ДЛЯ ИГРОКОВ (ВАРИАНТ 9)
# =============================================================================

# Для варианта 9 фиксируем конкретных агентов
player1_agents = np.array([4, 5, 6])  # Агенты первого игрока
player2_agents = np.array([1, 2, 3])  # Агенты второго игрока
neutral_agents = np.array([0, 7, 8, 9])  # Нейтральные агенты

print("\n" + "="*80)
print("3. РАСПРЕДЕЛЕНИЕ АГЕНТОВ (ВАРИАНТ 9):")
print(f"Агенты первого игрока (F): {player1_agents}")
print(f"Агенты второго игрока (S): {player2_agents}")
print(f"Нейтральные агенты: {neutral_agents}")

# =============================================================================
# 4. ВЫЧИСЛЕНИЕ ПАРАМЕТРОВ r_f, r_s, X^0
# =============================================================================

# Параметры из варианта 9
a, b, c, d = 4, 4, 1, 2
gf, gs = 2, 2

# Вычисляем r_f и r_s
r_f = np.sum(r[player1_agents])
r_s = np.sum(r[player2_agents])

# Генерируем начальные мнения для нейтральных агентов в диапазоне [0.4, 0.6] для баланса
np.random.seed(42)
neutral_opinions = np.random.uniform(0.4, 0.6, len(neutral_agents))

# Вычисляем X^0
X0 = np.sum(r[neutral_agents] * neutral_opinions)

print("\n" + "="*80)
print("4. ВЫЧИСЛЕНИЕ КЛЮЧЕВЫХ ПАРАМЕТРОВ:")
print(f"r_f = сумма r для агентов первого игрока = {r_f:.3f}")
print(f"r_s = сумма r для агентов второго игрока = {r_s:.3f}")
print(f"Нейтральные агенты: {neutral_agents}")
print(f"Их начальные мнения: {neutral_opinions.round(3)}")
print(f"X^0 = {X0:.3f}")

# =============================================================================
# 5. ОПРЕДЕЛЕНИЕ ФУНКЦИЙ ВЫИГРЫША И ЦЕЛЕВЫХ ФУНКЦИЙ
# =============================================================================

def final_opinion(u, v, r_f, r_s, X0):
    """Вычисляет итоговое мнение агентов: X(u,v) = r_f*u + r_s*v + X0"""
    return r_f * u + r_s * v + X0

def objective_f(u, v, r_f, r_s, X0, a, b, gf):
    """Целевая функция первого игрока: Φ_f(u,v) = a*X - b*X^2 - gf*u^2/2"""
    X = final_opinion(u, v, r_f, r_s, X0)
    return -(a * X - b * X**2 - gf * u**2 / 2)  # Минус для минимизации

def objective_s(u, v, r_f, r_s, X0, c, d, gs):
    """Целевая функция второго игрока: Φ_s(u,v) = c*X - d*X^2 - gs*v^2/2"""
    X = final_opinion(u, v, r_f, r_s, X0)
    return -(c * X - d * X**2 - gs * v**2 / 2)  # Минус для минимизации

# Оптимальные мнения (точки утопии)
X_max_f = a / (2 * b)  # Максимум H_f(x)
X_max_s = c / (2 * d)  # Максимум H_s(x)

print("\n" + "="*80)
print("5. ФУНКЦИИ ВЫИГРЫША И ОПТИМАЛЬНЫЕ МНЕНИЯ:")
print(f"H_f(x) = {a}*x - {b}*x^2")
print(f"H_s(x) = {c}*x - {d}*x^2")
print(f"Оптимальное мнение первого игрока X_max_f = a/(2b) = {X_max_f:.3f}")
print(f"Оптимальное мнение второго игрока X_max_s = c/(2d) = {X_max_s:.3f}")

# =============================================================================
# 6. ПОИСК РАВНОВЕСИЯ НЭША (УЛУЧШЕННАЯ ВЕРСИЯ)
# =============================================================================

def solve_analytical(r_f, r_s, X0, a, b, c, d, gf, gs):
    """Аналитическое решение системы уравнений с проверкой устойчивости"""
    
    print("\nРешение системы уравнений для равновесия Нэша:")
    print("∂Φ_f/∂u = a*r_f - 2*b*r_f*(r_f*u + r_s*v + X0) - gf*u = 0")
    print("∂Φ_s/∂v = c*r_s - 2*d*r_s*(r_f*u + r_s*v + X0) - gs*v = 0")
    print()
    
    # Коэффициенты системы уравнений:
    A11 = 2*b*r_f**2 + gf
    A12 = 2*b*r_f*r_s
    B1 = a*r_f - 2*b*r_f*X0
    
    A21 = 2*d*r_f*r_s
    A22 = 2*d*r_s**2 + gs
    B2 = c*r_s - 2*d*r_s*X0
    
    print(f"Система уравнений:")
    print(f"{A11:.3f}*u + {A12:.3f}*v = {B1:.3f}")
    print(f"{A21:.3f}*u + {A22:.3f}*v = {B2:.3f}")
    
    # Решаем систему
    A_matrix = np.array([[A11, A12], [A21, A22]])
    B_vector = np.array([B1, B2])
    
    # Проверяем условие устойчивости (определитель)
    det = A11 * A22 - A12 * A21
    print(f"Определитель матрицы: {det:.3f}")
    
    if abs(det) < 1e-10:
        print("Внимание: система близка к вырожденной!")
        # Используем псевдообратную матрицу
        solution = np.linalg.pinv(A_matrix) @ B_vector
    else:
        solution = np.linalg.solve(A_matrix, B_vector)
    
    u_opt, v_opt = solution[0], solution[1]
    
    # Применяем ограничения [0, 1] более мягко
    u_opt = np.clip(u_opt, 0, 1)
    v_opt = np.clip(v_opt, 0, 1)
    
    # Если решение на границе, пытаемся найти внутреннее решение численно
    if u_opt in [0, 1] or v_opt in [0, 1]:
        print("Решение на границе, используем численную оптимизацию...")
        u_opt, v_opt = find_numerical_equilibrium(r_f, r_s, X0, a, b, c, d, gf, gs)
    
    return u_opt, v_opt

def find_numerical_equilibrium(r_f, r_s, X0, a, b, c, d, gf, gs, max_iter=100, tol=1e-6):
    """Численный метод поиска равновесия Нэша"""
    
    # Начальные приближения
    u, v = 0.5, 0.5
    
    for iteration in range(max_iter):
        # Игрок 1 оптимизирует u при фиксированном v
        res1 = minimize(lambda u_val: objective_f(u_val, v, r_f, r_s, X0, a, b, gf), 
                       u, bounds=[(0, 1)], method='L-BFGS-B')
        u_new = res1.x[0]
        
        # Игрок 2 оптимизирует v при фиксированном u
        res2 = minimize(lambda v_val: objective_s(u_new, v_val, r_f, r_s, X0, c, d, gs), 
                       v, bounds=[(0, 1)], method='L-BFGS-B')
        v_new = res2.x[0]
        
        # Проверяем сходимость
        if abs(u_new - u) < tol and abs(v_new - v) < tol:
            print(f"Численная сходимость достигнута на итерации {iteration+1}")
            return u_new, v_new
        
        u, v = u_new, v_new
    
    print(f"Численный метод сошелся за {max_iter} итераций (точность: {abs(u_new - u):.2e}, {abs(v_new - v):.2e})")
    return u, v

# Находим равновесие Нэша
u_opt, v_opt = solve_analytical(r_f, r_s, X0, a, b, c, d, gf, gs)

print("\n" + "="*80)
print("6. РАВНОВЕСИЕ НЭША:")
print(f"Оптимальное управление первого игрока: u* = {u_opt:.3f}")
print(f"Оптимальное управление второго игрока: v* = {v_opt:.3f}")

# =============================================================================
# 7. ВЫЧИСЛЕНИЕ ИТОГОВОГО МНЕНИЯ И РАССТОЯНИЙ ДО ТОЧКИ УТОПИИ
# =============================================================================

# Итоговое мнение при оптимальных управлениях
X_final = final_opinion(u_opt, v_opt, r_f, r_s, X0)

# Расстояния до точек утопии
distance_f = abs(X_final - X_max_f)
distance_s = abs(X_final - X_max_s)

# Определяем победителя
winner = "Первый игрок" if distance_f < distance_s else "Второй игрок"

print("\n" + "="*80)
print("7. РЕЗУЛЬТАТЫ:")
print(f"Итоговое мнение агентов: X = {X_final:.3f}")
print(f"Точка утопии первого игрока: X_max_f = {X_max_f:.3f}")
print(f"Точка утопии второго игрока: X_max_s = {X_max_s:.3f}")
print(f"Расстояние до точки утопии первого игрока: Δ_f = {distance_f:.3f}")
print(f"Расстояние до точки утопии второго игрока: Δ_s = {distance_s:.3f}")
print(f"ПОБЕДИТЕЛЬ: {winner} (меньшее расстояние до точки утопии)")

# =============================================================================
# 8. УЛУЧШЕННЫЙ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ
# =============================================================================

def analyze_sensitivity(r_f, r_s, X0, X_max_f, X_max_s):
    """Расширенный анализ чувствительности"""
    
    print("\n" + "="*80)
    print("РАСШИРЕННЫЙ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ:")
    
    # Сценарий 1: Каждый игрок действует в одиночку
    X_solo_f = final_opinion(X_max_f, 0, r_f, r_s, X0)
    X_solo_s = final_opinion(0, X_max_s, r_f, r_s, X0)
    
    delta_solo_f = abs(X_solo_f - X_max_f)
    delta_solo_s = abs(X_solo_s - X_max_s)
    
    print(f"\n1. ДЕЙСТВИЯ В ОДИНОЧКУ:")
    print(f"   Первый игрок один: X = {X_solo_f:.3f}, Δ = {delta_solo_f:.3f}")
    print(f"   Второй игрок один: X = {X_solo_s:.3f}, Δ = {delta_solo_s:.3f}")
    
    # Сценарий 2: Кооперативное решение (Парето-оптимум)
    # Ищем u, v которые минимизируют сумму расстояний
    def cooperative_objective(x):
        u, v = x
        X = final_opinion(u, v, r_f, r_s, X0)
        return abs(X - X_max_f) + abs(X - X_max_s)
    
    res_coop = minimize(cooperative_objective, [0.5, 0.5], bounds=[(0,1), (0,1)])
    u_coop, v_coop = res_coop.x
    X_coop = final_opinion(u_coop, v_coop, r_f, r_s, X0)
    
    delta_coop_f = abs(X_coop - X_max_f)
    delta_coop_s = abs(X_coop - X_max_s)
    
    print(f"\n2. КООПЕРАТИВНОЕ РЕШЕНИЕ (Парето-оптимум):")
    print(f"   u_coop = {u_coop:.3f}, v_coop = {v_coop:.3f}")
    print(f"   X_coop = {X_coop:.3f}, Δ_f = {delta_coop_f:.3f}, Δ_s = {delta_coop_s:.3f}")
    
    # Сравнение выигрышей
    improvement_f = delta_solo_f - distance_f
    improvement_s = delta_solo_s - distance_s
    
    print(f"\n3. ВЫИГРЫШ ОТ КООПЕРАЦИИ (равновесие Нэша vs одиночная игра):")
    print(f"   Первый игрок: {improvement_f:+.3f} (улучшение)" if improvement_f > 0 else 
          f"   Первый игрок: {improvement_f:+.3f} (проигрыш)")
    print(f"   Второй игрок: {improvement_s:+.3f} (улучшение)" if improvement_s > 0 else 
          f"   Второй игрок: {improvement_s:+.3f} (проигрыш)")
    
    # Эффективность по Парето
    pareto_efficiency = (distance_f + distance_s) / (delta_coop_f + delta_coop_s)
    print(f"\n4. ЭФФЕКТИВНОСТЬ ПО ПАРЕТО:")
    print(f"   Эффективность: {pareto_efficiency:.2%}")

analyze_sensitivity(r_f, r_s, X0, X_max_f, X_max_s)

# =============================================================================
# 9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

# Создаем сетку значений для u и v в диапазоне [0, 1]
u_values = np.linspace(0, 1, 50)
v_values = np.linspace(0, 1, 50)
U, V = np.meshgrid(u_values, v_values)

# Вычисляем целевые функции на сетке
Z_f = np.zeros_like(U)
Z_s = np.zeros_like(U)
Z_X = np.zeros_like(U)

for i in range(len(u_values)):
    for j in range(len(v_values)):
        Z_f[j, i] = -objective_f(U[j, i], V[j, i], r_f, r_s, X0, a, b, gf)
        Z_s[j, i] = -objective_s(U[j, i], V[j, i], r_f, r_s, X0, c, d, gs)
        Z_X[j, i] = final_opinion(U[j, i], V[j, i], r_f, r_s, X0)

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# График целевой функции первого игрока
contour1 = axes[0,0].contourf(U, V, Z_f, levels=20, cmap='viridis')
axes[0,0].contour(U, V, Z_f, levels=10, colors='black', linewidths=0.5)
axes[0,0].plot(u_opt, v_opt, 'ro', markersize=10, label=f'Равновесие Нэша')
axes[0,0].set_xlabel('u (управление первого игрока)')
axes[0,0].set_ylabel('v (управление второго игрока)')
axes[0,0].set_title('Целевая функция первого игрока Φ_f(u,v)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
plt.colorbar(contour1, ax=axes[0,0])

# График целевой функции второго игрока
contour2 = axes[0,1].contourf(U, V, Z_s, levels=20, cmap='plasma')
axes[0,1].contour(U, V, Z_s, levels=10, colors='black', linewidths=0.5)
axes[0,1].plot(u_opt, v_opt, 'ro', markersize=10, label=f'Равновесие Нэша')
axes[0,1].set_xlabel('u (управление первого игрока)')
axes[0,1].set_ylabel('v (управление второго игрока)')
axes[0,1].set_title('Целевая функция второго игрока Φ_s(u,v)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
plt.colorbar(contour2, ax=axes[0,1])

# График итогового мнения
contour3 = axes[1,0].contourf(U, V, Z_X, levels=20, cmap='coolwarm')
axes[1,0].contour(U, V, Z_X, levels=10, colors='black', linewidths=0.5)
axes[1,0].plot(u_opt, v_opt, 'ro', markersize=10, label=f'Равновесие Нэша: X={X_final:.3f}')
axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label=f'X_max_f={X_max_f:.3f}')
axes[1,0].axvline(x=0, color='blue', linestyle='--', alpha=0.5, label=f'X_max_s={X_max_s:.3f}')
axes[1,0].set_xlabel('u (управление первого игрока)')
axes[1,0].set_ylabel('v (управление второго игрока)')
axes[1,0].set_title('Итоговое мнение агентов X(u,v)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
plt.colorbar(contour3, ax=axes[1,0])

# График расстояний до точек утопии
Z_dist = np.abs(Z_X - X_max_f) + np.abs(Z_X - X_max_s)
contour4 = axes[1,1].contourf(U, V, Z_dist, levels=20, cmap='hot')
axes[1,1].contour(U, V, Z_dist, levels=10, colors='white', linewidths=0.5)
axes[1,1].plot(u_opt, v_opt, 'ro', markersize=10, label=f'Равновесие Нэша')
axes[1,1].set_xlabel('u (управление первого игрока)')
axes[1,1].set_ylabel('v (управление второго игрока)')
axes[1,1].set_title('Суммарное расстояние до точек утопии')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
plt.colorbar(contour4, ax=axes[1,1])

plt.tight_layout()
plt.show()

# =============================================================================
# 10. ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ
# =============================================================================

print("\n" + "="*80)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ №6 (ВАРИАНТ 9)")
print("="*80)
print(f"Параметры варианта 9: a={a}, b={b}, c={c}, d={d}, gf={gf}, gs={gs}")
print(f"Агенты влияния: Player1={player1_agents}, Player2={player2_agents}")
print(f"Коэффициенты влияния: r_f={r_f:.3f}, r_s={r_s:.3f}")
print(f"Начальное мнение нейтральных агентов: X0={X0:.3f}")
print(f"Равновесие Нэша: u*={u_opt:.3f}, v*={v_opt:.3f}")
print(f"Итоговое мнение: X={X_final:.3f}")
print(f"Точки утопии: X_max_f={X_max_f:.3f}, X_max_s={X_max_s:.3f}")
print(f"Расстояния: Δ_f={distance_f:.3f}, Δ_s={distance_s:.3f}")
print(f"РЕЗУЛЬТАТ: {winner}")

# Проверка корректности решения
print("\n" + "="*80)
print("ПРОВЕРКА КОРРЕКТНОСТИ РЕШЕНИЯ:")
print("1. Матрица A стохастическая:", np.allclose(np.sum(A, axis=1), 1.0, atol=1e-10))
print("2. Сумма элементов вектора r равна 1:", np.isclose(r.sum(), 1.0, atol=1e-10))
print("3. Все строки A∞ одинаковы:", np.allclose(A_inf, np.tile(r, (n_agents, 1)), atol=1e-10))
print("4. Управления в допустимом диапазоне [0,1]:", 
      f"u*={u_opt:.3f} ∈ [0,1]: {0 <= u_opt <= 1}, "
      f"v*={v_opt:.3f} ∈ [0,1]: {0 <= v_opt <= 1}")