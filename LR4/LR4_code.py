import random
from typing import List, Tuple, Optional
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, red, blue, green
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

class GameNode:
    """Класс, представляющий узел в дереве позиционной игры."""
    def __init__(self, node_id: int, is_terminal: bool = False, payoffs: Tuple[int, int] = None, 
                 depth: int = 0, player: str = None):
        self.id = node_id
        self.is_terminal = is_terminal
        self.payoffs = payoffs if payoffs else (0, 0)
        self.children = []
        self.depth = depth
        self.player = player
        self.optimum_payoffs = None
        self.optimal_children = []
        self.x = 0
        self.y = 0

def generate_game_tree(depth: int, num_players: int, branches_per_player: List[int], 
                      payoff_range: Tuple[int, int], current_depth: int = 0, 
                      node_counter: list = [0]) -> Optional[GameNode]:
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

    node = GameNode(node_id=node_id, is_terminal=is_terminal, 
                   depth=current_depth, player=current_player)

    if is_terminal:
        payoff_a = random.randint(payoff_range[0], payoff_range[1])
        payoff_b = random.randint(payoff_range[0], payoff_range[1])
        node.payoffs = (payoff_a, payoff_b)
        node.optimum_payoffs = (payoff_a, payoff_b)
    else:
        for _ in range(num_branches):
            child = generate_game_tree(depth, num_players, branches_per_player, 
                                      payoff_range, current_depth + 1, node_counter)
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

def calculate_positions(node: GameNode, x_spacing: float = 2.0, y_spacing: float = 2.0, x_offset: float = 0.0):
    """
    Вычисляет позиции для визуализации дерева.
    """
    if not node.children:
        node.x = x_offset
        node.y = -node.depth * y_spacing
        return 1, x_offset
    
    total_width = 0
    current_x = x_offset
    
    for child in node.children:
        child_width, new_x = calculate_positions(child, x_spacing, y_spacing, current_x)
        total_width += child_width
        current_x = new_x + x_spacing
    
    node.x = x_offset + total_width / 2
    node.y = -node.depth * y_spacing
    
    return total_width, current_x

def visualize_tree(root: GameNode, optimal_paths: List[List[GameNode]]):
    """
    Создает визуализацию дерева с помощью NetworkX и Matplotlib.
    """
    G = nx.DiGraph()
    pos = {}
    labels = {}
    node_colors = []
    node_sizes = []
    
    # Собираем все узлы в граф
    def add_nodes_edges(node):
        pos[node.id] = (node.x, node.y)
        
        if node.is_terminal:
            labels[node.id] = f"{node.id}\n{node.payoffs}"
            node_colors.append('lightgreen')
            node_sizes.append(800)
        else:
            labels[node.id] = f"{node.id}\n{node.player}\n{node.optimum_payoffs}"
            node_colors.append('lightblue')
            node_sizes.append(1000)
        
        for child in node.children:
            G.add_edge(node.id, child.id)
            add_nodes_edges(child)
    
    add_nodes_edges(root)
    
    # Определяем цвет ребер (красный для оптимальных путей)
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        is_optimal = False
        for path in optimal_paths:
            path_ids = [n.id for n in path]
            for i in range(len(path_ids) - 1):
                if path_ids[i] == u and path_ids[i+1] == v:
                    is_optimal = True
                    break
            if is_optimal:
                break
        edge_colors.append('red' if is_optimal else 'gray')
        edge_widths.append(2.0 if is_optimal else 1.0)
    
    # Создаем визуализацию
    plt.figure(figsize=(30, 15))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, 
            edge_color=edge_colors, node_size=node_sizes, arrows=True, 
            width=edge_widths, alpha=0.7)
    
    # Рисуем метки с улучшенным форматированием
    for node_id, (x, y) in pos.items():
        label = labels[node_id]
        plt.text(x, y, label, fontsize=6, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.title("Дерево позиционной игры", fontsize=16)
    plt.axis('off')
    
    # Сохраняем в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def create_pdf_report(root: GameNode, optimal_paths: List[List[GameNode]], filename: str = "game_tree_report.pdf"):
    """
    Создает PDF-отчет с визуализацией дерева и информацией о оптимальных путях.
    """
    # Регистрируем шрифт с поддержкой кириллицы
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        font_name = 'DejaVuSans'
    except:
        font_name = 'Helvetica'
        print("Шрифт DejaVuSans не найден, используется Helvetica (кириллица может отображаться некорректно)")
    
    # Вычисляем позиции для визуализации
    calculate_positions(root, x_spacing=3.0, y_spacing=2.0)
    
    # Создаем визуализацию дерева
    tree_image = visualize_tree(root, optimal_paths)
    
    # Создаем PDF
    c = canvas.Canvas(filename, pagesize=landscape(A4))
    width, height = landscape(A4)
    
    # Добавляем заголовок
    c.setFont(font_name, 16)
    c.drawString(50, height - 50, "Отчет по дереву позиционной игры")
    
    # Добавляем информацию о параметрах
    c.setFont(font_name, 12)
    c.drawString(50, height - 80, f"Глубина дерева: 7")
    c.drawString(50, height - 100, f"Количество игроков: 2 (A и B)")
    c.drawString(50, height - 120, f"Количество стратегий: 2 для каждого игрока")
    c.drawString(50, height - 140, f"Диапазон выигрышей: [0, 20]")
    c.drawString(50, height - 160, f"Оптимальные выигрыши в корне: {root.optimum_payoffs}")
    c.drawString(50, height - 180, f"Количество оптимальных путей: {len(optimal_paths)}")
    
    # Добавляем изображение дерева
    img = ImageReader(tree_image)
    img_width, img_height = img.getSize()
    aspect = img_height / img_width
    
    # Размещаем изображение по центру
    display_width = width - 100
    display_height = display_width * aspect
    
    # Проверяем, помещается ли изображение по высоте
    if display_height > height - 250:
        display_height = height - 250
        display_width = display_height / aspect
    
    c.drawImage(img, (width - display_width) / 2, height - 250 - display_height, 
                width=display_width, height=display_height)
    
    # Добавляем информацию об оптимальных путях
    y_position = height - 250 - display_height - 30
    c.setFont(font_name, 12)
    c.drawString(50, y_position, "Оптимальные пути:")
    
    c.setFont(font_name, 10)
    y_position -= 20
    
    for i, path in enumerate(optimal_paths):
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            c.setFont(font_name, 10)
        
        path_ids = [f"{n.id}" for n in path]
        terminal_payoffs = path[-1].payoffs
        path_str = " → ".join(path_ids)
        c.drawString(70, y_position, f"Путь {i+1}: {path_str} → Выигрыши: {terminal_payoffs}")
        y_position -= 15
    
    # Добавляем легенду
    c.showPage()
    c.setFont(font_name, 14)
    c.drawString(50, height - 50, "Легенда:")
    
    c.setFont(font_name, 12)
    c.drawString(50, height - 80, "Синие узлы: нетерминальные вершины (с указанием игрока и оптимальных выигрышей)")
    c.drawString(50, height - 100, "Зеленые узлы: терминальные вершины (с указанием выигрышей)")
    c.drawString(50, height - 120, "Красные ребра: оптимальные пути")
    c.drawString(50, height - 140, "Серые ребра: неоптимальные пути")
    
    # Сохраняем PDF
    c.save()

def main():
    """Основная функция, выполняющая лабораторную работу."""
    # Параметры варианта 9
    DEPTH = 7
    NUM_PLAYERS = 2
    BRANCHES_PER_PLAYER = [2, 2]
    PAYOFF_RANGE = (0, 20)

    print("Генерация дерева игры...")
    node_counter = [0]
    root_node = generate_game_tree(
        depth=DEPTH,
        num_players=NUM_PLAYERS,
        branches_per_player=BRANCHES_PER_PLAYER,
        payoff_range=PAYOFF_RANGE,
        current_depth=0,
        node_counter=node_counter
    )
    print(f"Дерево сгенерировано. Всего узлов: {node_counter[0]}")
    print(f"Количество терминальных узлов (листьев): {2**DEPTH}")

    print("\nВыполнение обратной индукции...")
    backward_induction(root_node)
    print("Обратная индукция завершена.")

    print(f"\nОптимальные выигрыши в корневой вершине (игрок A, игрок B): {root_node.optimum_payoffs}")

    print("\nПоиск всех оптимальных путей...")
    optimal_paths = find_optimal_paths(root_node)
    print(f"Найдено оптимальных путей: {len(optimal_paths)}")

    print("\nСоздание PDF-отчета...")
    create_pdf_report(root_node, optimal_paths, "game_tree_report.pdf")
    print("PDF-отчет создан: game_tree_report.pdf")

if __name__ == "__main__":
    main()