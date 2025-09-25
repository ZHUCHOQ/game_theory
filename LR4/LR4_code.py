import random
from typing import List, Tuple, Optional
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import matplotlib

# Set font to avoid any language issues
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class GameNode:
    """Class representing a node in the positional game tree."""
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
    Recursively generates the game tree.
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
    Performs backward induction on the game tree.
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
    Finds all optimal paths from root to terminal nodes.
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

def calculate_horizontal_positions(root: GameNode, x_spacing: float = 2.0, y_spacing: float = 1.0):
    """
    Calculates positions for a horizontal tree (root on left, leaves on right).
    Uses BFS for uniform node distribution.
    """
    # Collect nodes by depth
    levels = {}
    queue = [(root, 0)]  # (node, vertical position)
    
    while queue:
        node, pos = queue.pop(0)
        if node.depth not in levels:
            levels[node.depth] = []
        levels[node.depth].append((node, pos))
        
        # Distribute children
        if node.children:
            child_count = len(node.children)
            start_pos = pos - (child_count - 1) / 2
            for i, child in enumerate(node.children):
                child_pos = start_pos + i
                queue.append((child, child_pos))
    
    # Calculate actual coordinates
    max_depth = max(levels.keys())
    max_width = max(len(nodes) for nodes in levels.values())
    
    for depth, nodes in levels.items():
        # Sort nodes by vertical position
        nodes.sort(key=lambda x: x[1])
        
        # Distribute evenly vertically
        for i, (node, _) in enumerate(nodes):
            node.x = depth * x_spacing
            node.y = (i - len(nodes)/2) * y_spacing

def visualize_horizontal_tree(root: GameNode, optimal_paths: List[List[GameNode]]):
    """
    Creates a horizontal tree visualization (root on left, leaves on right).
    """
    G = nx.DiGraph()
    pos = {}
    labels = {}
    node_colors = []
    node_sizes = []
    
    # Collect all nodes into graph
    def add_nodes_edges(node):
        pos[node.id] = (node.x, node.y)
        
        if node.is_terminal:
            labels[node.id] = f"{node.id}\n{node.payoffs}"
            node_colors.append('lightgreen')
            node_sizes.append(500)
        else:
            labels[node.id] = f"{node.id}\n{node.player}\n{node.optimum_payoffs}"
            node_colors.append('lightblue')
            node_sizes.append(700)
        
        for child in node.children:
            G.add_edge(node.id, child.id)
            add_nodes_edges(child)
    
    add_nodes_edges(root)
    
    # Determine edge colors (red for optimal paths)
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
    
    # Create horizontal visualization
    plt.figure(figsize=(20, 12))
    
    # Draw graph with horizontal orientation
    nx.draw(G, pos, with_labels=False, node_color=node_colors, 
            edge_color=edge_colors, node_size=node_sizes, arrows=True, 
            width=edge_widths, alpha=0.7, arrowsize=15)
    
    # Draw labels with improved formatting
    for node_id, (x, y) in pos.items():
        label = labels[node_id]
        plt.text(x, y, label, fontsize=5, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.title("Positional Game Tree", fontsize=14)
    plt.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def create_pdf_report(root: GameNode, optimal_paths: List[List[GameNode]], filename: str = "game_tree_report.pdf"):
    """
    Creates PDF report with horizontal tree visualization.
    """
    # Calculate positions for horizontal visualization
    calculate_horizontal_positions(root, x_spacing=1.5, y_spacing=1.2)
    
    # Create horizontal tree visualization
    tree_image = visualize_horizontal_tree(root, optimal_paths)
    
    # Create PDF
    c = canvas.Canvas(filename, pagesize=landscape(A4))
    width, height = landscape(A4)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Positional Game Tree Analysis")
    
    # Add parameters information
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Tree depth: 7")
    c.drawString(50, height - 100, f"Number of players: 2 (A and B)")
    c.drawString(50, height - 120, f"Strategies per player: 2")
    c.drawString(50, height - 140, f"Payoff range: [0, 20]")
    c.drawString(50, height - 160, f"Optimal payoffs at root: {root.optimum_payoffs}")
    c.drawString(50, height - 180, f"Number of optimal paths: {len(optimal_paths)}")
    
    # Add tree image - use full page width
    img = ImageReader(tree_image)
    img_width, img_height = img.getSize()
    aspect = img_height / img_width
    
    # Use full page width for the tree
    display_width = width - 100
    display_height = display_width * aspect
    
    # Adjust height if needed
    max_display_height = height - 250
    if display_height > max_display_height:
        display_height = max_display_height
        display_width = display_height / aspect
    
    # Center the image
    x_offset = (width - display_width) / 2
    y_offset = height - 250 - display_height
    
    c.drawImage(img, x_offset, y_offset, width=display_width, height=display_height)
    
    # Add optimal paths information
    y_position = y_offset - 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Optimal paths:")
    
    c.setFont("Helvetica", 10)
    y_position -= 20
    
    for i, path in enumerate(optimal_paths):
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, "Optimal paths (continued):")
            y_position -= 20
        
        path_ids = [f"{n.id}" for n in path]
        terminal_payoffs = path[-1].payoffs
        path_str = " -> ".join(path_ids)
        c.drawString(70, y_position, f"Path {i+1}: {path_str} -> Payoffs: {terminal_payoffs}")
        y_position -= 15
    
    # Add legend
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Legend:")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, "Blue nodes: non-terminal vertices (player and optimal payoffs)")
    c.drawString(50, height - 100, "Green nodes: terminal vertices (payoffs)")
    c.drawString(50, height - 120, "Red edges: optimal paths")
    c.drawString(50, height - 140, "Gray edges: non-optimal paths")
    c.drawString(50, height - 160, "Orientation: root on left, leaves on right")
    
    # Save PDF
    c.save()

def main():
    """Main function performing the laboratory work."""
    # Parameters for variant 9
    DEPTH = 7
    NUM_PLAYERS = 2
    BRANCHES_PER_PLAYER = [2, 2]
    PAYOFF_RANGE = (0, 20)

    print("Generating game tree...")
    node_counter = [0]
    root_node = generate_game_tree(
        depth=DEPTH,
        num_players=NUM_PLAYERS,
        branches_per_player=BRANCHES_PER_PLAYER,
        payoff_range=PAYOFF_RANGE,
        current_depth=0,
        node_counter=node_counter
    )
    print(f"Tree generated. Total nodes: {node_counter[0]}")
    print(f"Terminal nodes (leaves): {2**DEPTH}")

    print("\nPerforming backward induction...")
    backward_induction(root_node)
    print("Backward induction completed.")

    print(f"\nOptimal payoffs at root (player A, player B): {root_node.optimum_payoffs}")

    print("\nFinding all optimal paths...")
    optimal_paths = find_optimal_paths(root_node)
    print(f"Optimal paths found: {len(optimal_paths)}")

    print("\nCreating PDF report with horizontal tree...")
    create_pdf_report(root_node, optimal_paths, "game_tree_report.pdf")
    print("PDF report created: game_tree_report.pdf")

if __name__ == "__main__":
    main()