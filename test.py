import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def draw_regular_polygons_with_labels(loop, choices):
    # Set up the figure
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Offset for separating polygons
    offset_x = 5
    offset_y = 5

    # Store node coordinates
    points_coordinates = {}

    # Draw polygons
    for idx, points in enumerate(loop):
        sides = len(points)
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)

        radius = 2
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        x_offset = idx * offset_x
        y_offset = idx * offset_y

        x_translated = x + x_offset
        y_translated = y + y_offset

        for i in range(sides):
            points_coordinates[points[i]] = (x_translated[i], y_translated[i])

        # Draw polygon edges
        for i in range(sides):
            x1, y1 = x_translated[i], y_translated[i]
            x2, y2 = x_translated[(i + 1) % sides], y_translated[(i + 1) % sides]
            arrow = FancyArrowPatch((x1, y1), (x2, y2), mutation_scale=15, color="blue", arrowstyle='->', lw=2)
            ax.add_patch(arrow)

        # Draw nodes
        for i in range(sides):
            ax.plot(x_translated[i], y_translated[i], 'bo', markersize=20, markerfacecolor='none', markeredgewidth=2)
            ax.text(x_translated[i], y_translated[i], points[i], fontsize=12, ha='center', va='center', color='black')

    # Draw choices with visited set to prevent infinite recursion
    visited = set()

    def draw_choices(node, target_node, radius=2.8):
        if node in visited:
            return
        visited.add(node)

        if target_node in points_coordinates:
            x_target, y_target = points_coordinates[target_node]

            pointing_nodes = [key for key, value in choices.items() if value == target_node]
            num_choices = len(pointing_nodes)

            if num_choices > 0:
                angles = np.linspace(0, 2 * np.pi, num_choices, endpoint=False)

                for i, pointing_node in enumerate(pointing_nodes):
                    angle = angles[i]
                    x_choice = x_target + radius * np.cos(angle)
                    y_choice = y_target + radius * np.sin(angle)

                    ax.plot(x_choice, y_choice, 'go', markersize=12, markerfacecolor='none', markeredgewidth=2)
                    ax.text(x_choice, y_choice, str(pointing_node), fontsize=12, ha='center', va='center', color='black')

                    arrow = FancyArrowPatch((x_choice, y_choice), (x_target, y_target), mutation_scale=15, color="red", arrowstyle='->', lw=2)
                    ax.add_patch(arrow)

                    draw_choices(pointing_node, target_node, radius=radius + 1.5)

    for node, target_node in choices.items():
        draw_choices(node, target_node)

    ax.grid(True)
    plt.title("Multiple Regular Polygons with Node Labels and Choices")
    plt.show()

# Example polygons
loop = [
    [1, 2, 3],  # Triangle
    ['4', '5', '6', '7'],  # Square
    ['H', 'I', 'J', 'K', 'L'],  # Pentagon
    ['M', 'N', 'O', 'P', 'Q', 'R']  # Hexagon
]

# Example choices
choices = {
    10: 1,  # Node 10 points to node 1
    11: 1,
    12: 1,
    22: 12,
    33: 12
}

draw_regular_polygons_with_labels(loop, choices)
