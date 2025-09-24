import numpy as np
from random import randint

# Параметры задачи - 10 покупателей
N = 10
RANGE = (0, 20000)
ROUND_DIGITS = 5

print("=" * 60)
print("АУКЦИОН ПЕРВОЙ ЦЕНЫ ДЛЯ 10 ПОКУПАТЕЛЕЙ")
print("=" * 60)

# Генерация оценок товара для 10 покупателей
X_VALUES = [randint(*RANGE) for _ in range(N)]

print(f"\nОЦЕНКИ ТОВАРА ДЛЯ {N} ПОКУПАТЕЛЕЙ:")
for i, value in enumerate(X_VALUES):
    print(f"Покупатель {i+1}: {value}")

print(f"\nРАВНОВЕСНАЯ СТРАТЕГИЯ ДЛЯ {N} УЧАСТНИКОВ:")
print(f"b(x) = ({N}-1)*x/{N} = {(N-1)}*x/{N}")

# Расчет оптимальных ставок для каждого покупателя
bets = []
print("\nРАСЧЕТ ОПТИМАЛЬНЫХ СТАВОК:")
print("-" * 40)

for i in range(N):
    # Формула равновесной стратегии: (n-1)*x/n
    bet = (N - 1) * X_VALUES[i] / N
    bets.append(bet)
    
    print(f"Покупатель {i+1}:")
    print(f"  Оценка товара: {X_VALUES[i]}")
    print(f"  Расчет ставки: ({N}-1)*{X_VALUES[i]}/{N} = {bet:.{ROUND_DIGITS}f}")

# Определение победителя
winner_index = np.argmax(bets)
winner_value = X_VALUES[winner_index]
winner_bet = bets[winner_index]
winner_score = winner_value - winner_bet

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ АУКЦИОНА:")
print("=" * 60)

print(f"Победитель: покупатель {winner_index + 1}")
print(f"Его оценка товара: {winner_value}")
print(f"Его ставка: {winner_bet:.{ROUND_DIGITS}f}")
print(f"Выигрыш победителя: {winner_value} - {winner_bet:.{ROUND_DIGITS}f} = {winner_score:.{ROUND_DIGITS}f}")
print(f"Цена игры (ставка победителя): {winner_bet:.{ROUND_DIGITS}f}")

# Дополнительная информация о всех участниках
print("\nДЕТАЛЬНАЯ ИНФОРМАЦИЯ ОБ УЧАСТНИКАХ:")
print("-" * 60)

for i in range(N):
    status = "★ ПОБЕДИТЕЛЬ" if i == winner_index else ""
    print(f"Покупатель {i+1}: оценка = {X_VALUES[i]:>6}, ставка = {bets[i]:>{ROUND_DIGITS+7}.{ROUND_DIGITS}f} {status}")

# Статистика
print("\nСТАТИСТИКА:")
print("-" * 60)
print(f"Максимальная оценка: {max(X_VALUES)}")
print(f"Минимальная оценка: {min(X_VALUES)}")
print(f"Средняя оценка: {np.mean(X_VALUES):.{ROUND_DIGITS}f}")
print(f"Максимальная ставка: {max(bets):.{ROUND_DIGITS}f}")
print(f"Минимальная ставка: {min(bets):.{ROUND_DIGITS}f}")
print(f"Средняя ставка: {np.mean(bets):.{ROUND_DIGITS}f}")

print("\n" + "=" * 60)
print("ТЕОРЕТИЧЕСКОЕ ОБОСНОВАНИЕ:")
print("=" * 60)
print("В аукционе первой цены с независимыми частными оценками")
print(f"равновесная стратегия для каждого участника: b(x) = ({N}-1)/{N} * x")
print("где x - субъективная оценка товара участником")
print("Победитель платит свою ставку и получает выигрыш = оценка - ставка")