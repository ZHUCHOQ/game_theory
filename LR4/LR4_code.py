import random
from typing import List, Tuple, Optional

class GameNode:
    """Класс, представляющий узел в дереве позиционной игры."""
    def __init__(self, node_id: int, is_terminal: bool = False, payoffs: Tuple[int, int] = None, depth: int = 0, player: str = None):
        self.id = node_id
        self.is_terminal = is_terminal
        self.payoffs = payoffs if payoffs else (0, 0)
        self.children = []
        self.depth = depth
        self.player = player
        self.optimum_payoffs = None
        self.optimal_children = []

def generate_game_tree(depth: int, num_players: int, branches_per_player: List[int], payoff_range: Tuple[int, int], current_depth: int, node_counter: list) -> Optional['GameNode']:
    """
    Рекурсивно генерирует дерево игры.
    """
    if current_depth > depth:
        return None

    node_id = node_counter[0]
    node_counter[0] += 1
    is_terminal = (current_depth == depth)
    current_player_index = current_depth % num_players
    current_player = chr(65 + current_player_index)
    num_branches = branches_per_player[current_player_index]

    node = GameNode(node_id=node_id, is_terminal=is_terminal, depth=current_depth, player=current_player)

    if is_terminal:
        payoff_a = random.randint(payoff_range[0], payoff_range[1])
        payoff_b = random.randint(payoff_range[0], payoff_range[1])
        node.payoffs = (payoff_a, payoff_b)
        node.optimum_payoffs = (payoff_a, payoff_b)
    else:
        for _ in range(num_branches):
            child = generate_game_tree(depth, num_players, branches_per_player, payoff_range, current_depth + 1, node_counter)
            if child is not None:
                node.children.append(child)
    return node

def backward_induction(node: GameNode):
    """
    Выполняет алгоритм обратной индукции на дереве игры.
    """
    if node.is_terminal:
        return

    for child in node.children:
        backward_induction(child)

    optimizing_player_index = 0 if node.player == 'A' else 1
    best_value = -10**9
    best_children = []

    for child in node.children:
        child_payoff = child.optimum_payoffs[optimizing_player_index]
        if child_payoff > best_value:
            best_value = child_payoff
            best_children = [child]
        elif child_payoff == best_value:
            best_children.append(child)

    if best_children:
        node.optimal_children = best_children
        node.optimum_payoffs = best_children[0].optimum_payoffs

def find_optimal_paths(root: GameNode) -> List[List[GameNode]]:
    """
    Находит все оптимальные пути от корня до терминальных узлов.
    """
    all_paths = []
    current_path = []

    def dfs_collect_paths(current_node):
        current_path.append(current_node)
        if current_node.is_terminal:
            all_paths.append(current_path[:])
        else:
            for optimal_child in current_node.optimal_children:
                dfs_collect_paths(optimal_child)
        current_path.pop()

    dfs_collect_paths(root)
    return all_paths

def print_tree(node: GameNode, max_depth: int = 3, current_depth: int = 0):
    """
    Рекурсивно печатает дерево в виде структурированного текста.
    """
    if current_depth > max_depth:
        return
    indent = "  " * current_depth
    node_type = "T" if node.is_terminal else "N"
    payoffs_str = f" Payoffs: {node.payoffs}" if node.is_terminal else ""
    optimum_str = f" Optimum: {node.optimum_payoffs}" if node.optimum_payoffs is not None else ""
    print(f"{indent}{node_type}{node.id} (P{node.player}{payoffs_str}{optimum_str})")
    for child in node.children:
        print_tree(child, max_depth, current_depth + 1)

def main():
    DEPTH = 7
    NUM_PLAYERS = 2
    BRANCHES_PER_PLAYER = [2, 2]
    PAYOFF_RANGE = (0, 20)

    node_counter = [0]
    root_node = generate_game_tree(
        depth=DEPTH,
        num_players=NUM_PLAYERS,
        branches_per_player=BRANCHES_PER_PLAYER,
        payoff_range=PAYOFF_RANGE,
        current_depth=0,
        node_counter=node_counter
    )

    print("Генерация дерева игры...")
    print(f"Дерево сгенерировано. Всего узлов: {node_counter[0]}")
    print(f"Количество терминальных узлов (листьев): {2**DEPTH}")

    print("\nВыполнение обратной индукции...")
    backward_induction(root_node)
    print("Обратная индукция завершена.")

    print(f"\nОптимальные выигрыши в корневой вершине (игрок A, игрок B): {root_node.optimum_payoffs}")

    print("\nПоиск всех оптимальных путей...")
    optimal_paths = find_optimal_paths(root_node)
    print(f"Найдено оптимальных путей: {len(optimal_paths)}")

    print("\nПримеры оптимальных путей (первые 3 или меньше):")
    for i, path in enumerate(optimal_paths[:3]):
        path_ids = [n.id for n in path]
        terminal_payoffs = path[-1].payoffs
        print(f"Путь {i+1}: {path_ids} -> Выигрыши в конечной вершине: {terminal_payoffs}")
    if len(optimal_paths) > 3:
        print(f"... и еще {len(optimal_paths) - 3} путей.")

    print("\n\nВизуализация структуры дерева (ограничена глубиной 3 для читаемости):")
    print_tree(root_node, max_depth=3)

if __name__ == "__main__":
    main()