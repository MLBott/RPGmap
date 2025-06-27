import numpy as np
import noise
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from scipy.ndimage import zoom
import random 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog
import json
import sys


class Node:
    """
    A class to represent a single point on the map. It holds all the
    attributes for a specific coordinate, making the map data easier to manage.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.elevation = 0.0  # Raw height value, normalized 0-1
        self.moisture = 0.0   # Raw moisture value, normalized 0-1
        
        # Final assigned terrain properties
        self.terrain_type = ' ' # Single-letter code (e.g., 'M' for Mountain)
        self.terrain_label = '' # Descriptive name (e.g., "Rocky Mountain")
        self.is_river = False

    def __repr__(self):
        """String representation for printing node details."""
        return (f"Node({self.x}, {self.y}):\n"
                f"  Elevation: {self.elevation:.2f}\n"
                f"  Moisture: {self.moisture:.2f}\n"
                f"  Type: '{self.terrain_type}' ({self.terrain_label})")

class MapGenerator:
    """
    Manages the procedural generation of the entire world map.
    This class handles noise generation, erosion, and terrain classification.
    """
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else np.random.randint(0, 1000)
        
        # Initialize a 2D grid of Node objects
        self.grid = [[Node(x, y) for x in range(width)] for y in range(height)]

        # --- Generation Parameters ---
        # Noise settings control the overall shape of the terrain
        self.elevation_scale = 40.0
        self.elevation_octaves = 6
        self.elevation_persistence = 0.5
        self.elevation_lacunarity = 2.0

        self.moisture_scale = 50.0
        self.moisture_octaves = 4
        self.moisture_persistence = 0.5
        self.moisture_lacunarity = 2.0

        # Terrain thresholds control biome distribution (0-1 range)
        self.sea_level = 0.30
        self.beach_level = 0.32
        self.forest_level = 0.50
        self.mountain_level = 0.70
        self.snow_level = 0.90
        self.cliff_threshold = 0.08 # Gradient threshold for a cliff

    def generate_world(self):
        """Executes all steps to create the world from scratch."""
        print(f"Generating world with seed: {self.seed}")
        self._generate_noise_maps()
        self._create_rivers()
        self._classify_terrain()
        self._detect_cliffs()
        print("World generation complete.")
        return self.grid

    def _generate_noise_maps(self):
        """Generates and normalizes elevation and moisture maps."""
        print("Step 1: Generating noise maps...")
        elevation_data = np.zeros((self.height, self.width))
        moisture_data = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                elevation_data[y][x] = noise.pnoise2(x / self.elevation_scale, 
                                                     y / self.elevation_scale,
                                                     octaves=self.elevation_octaves,
                                                     persistence=self.elevation_persistence,
                                                     lacunarity=self.elevation_lacunarity,
                                                     base=self.seed)
                
                moisture_data[y][x] = noise.pnoise2(x / self.moisture_scale, 
                                                    y / self.moisture_scale,
                                                    octaves=self.moisture_octaves,
                                                    persistence=self.moisture_persistence,
                                                    lacunarity=self.moisture_lacunarity,
                                                    base=self.seed + 1) # Use a different seed for moisture

        # Normalize maps to a 0-1 range
        elevation_data = (elevation_data - np.min(elevation_data)) / (np.max(elevation_data) - np.min(elevation_data))
        moisture_data = (moisture_data - np.min(moisture_data)) / (np.max(moisture_data) - np.min(moisture_data))

        # Assign values to the nodes
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x].elevation = elevation_data[y][x]
                self.grid[y][x].moisture = moisture_data[y][x]

    def _create_rivers(self, num_rivers=15, max_length=200, min_length=10):
        """Simulates river formation using a steepest-descent path."""
        print("Step 2: Carving rivers...")
        rivers_created = 0
        # Try more times than num_rivers, as some attempts will fail
        for _ in range(num_rivers * 10): 
            if rivers_created >= num_rivers:
                break

            # Find a random starting point in a high-elevation area
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.grid[y][x].elevation < self.mountain_level:
                continue
            
            # --- 1. Trace the potential river path without modifying the grid ---
            path = []
            current_x, current_y = x, y
            
            for _ in range(max_length):
                # Stop if we flow into an existing river or the sea
                if self.grid[current_y][current_x].is_river or self.grid[current_y][current_x].elevation < self.sea_level:
                    break
                
                path.append((current_y, current_x))
                
                # Find the neighbor with the lowest elevation
                neighbors = []
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0: continue
                        nx, ny = current_x + dx, current_y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            neighbors.append((ny, nx))
                
                if not neighbors: break

                next_y, next_x = min(neighbors, key=lambda p: self.grid[p[0]][p[1]].elevation)
                
                # Stop if we're in a pit (local minima)
                if self.grid[next_y][next_x].elevation >= self.grid[current_y][current_x].elevation:
                    break
                    
                current_x, current_y = next_x, next_y
            
            # --- 2. If the path is long enough, apply it to the grid ---
            if len(path) >= min_length:
                rivers_created += 1
                for ry, rx in path:
                    node = self.grid[ry][rx]
                    node.is_river = True
                    # Lower elevation to "carve" the riverbed
                    node.elevation = max(self.sea_level - 0.05, node.elevation * 0.85)


    def _classify_terrain(self):
        """Assigns a terrain type and label to each node based on its properties."""
        print("Step 3: Classifying terrain biomes...")
        for y in range(self.height):
            for x in range(self.width):
                node = self.grid[y][x]
                
                # River classification must come first
                if node.is_river:
                    node.terrain_type = 'R'
                    node.terrain_label = 'River'
                    continue

                if node.elevation < self.sea_level:
                    node.terrain_type = 'D'
                    node.terrain_label = 'Deep Water'
                elif node.elevation < self.beach_level:
                    node.terrain_type = 'S'
                    node.terrain_label = 'Sandy Beach'
                elif node.elevation >= self.snow_level:
                    node.terrain_type = 'P'
                    node.terrain_label = 'Snowy Peak'
                elif node.elevation >= self.mountain_level:
                    node.terrain_type = 'M'
                    node.terrain_label = 'Rocky Mountain'
                elif node.elevation >= self.forest_level:
                    if node.moisture > 0.4:
                        node.terrain_type = 'F'
                        node.terrain_label = 'Dense Forest'
                    else:
                        node.terrain_type = 'H'
                        node.terrain_label = 'Highlands'
                else: # Plains and Grasslands
                    if node.moisture > 0.5:
                        node.terrain_type = 'f'
                        node.terrain_label = 'Forested Plains'
                    elif node.moisture > 0.3:
                        node.terrain_type = 'p'
                        node.terrain_label = 'Plains'
                    else:
                        node.terrain_type = 'g'
                        node.terrain_label = 'Grassland'
        for y in range(self.height):
            for x in range(self.width):
                node = self.grid[y][x]
                # Impassable border
                if x == 0 or y == 0 or x == self.width-1 or y == self.height-1:
                    node.terrain_type = 'X'
                    node.terrain_label = 'Impassable'
                    continue

    def _detect_cliffs(self):
        """A second pass to identify steep areas as cliffs."""
        print("Step 4: Detecting cliffs...")
        for y in range(1, self.height):
            for x in range(1, self.width):
                node = self.grid[y][x]
                # Don't create cliffs on rivers
                if node.is_river:
                    continue
                if node.elevation > self.beach_level:
                    # Calculate gradient
                    dx = node.elevation - self.grid[y][x-1].elevation
                    dy = node.elevation - self.grid[y-1][x].elevation
                    gradient = np.sqrt(dx*dx + dy*dy)
                    
                    if gradient > self.cliff_threshold:
                        node.terrain_type = 'C'
                        node.terrain_label = 'Steep Cliff Face'

    def add_castle_region(self, size=70, terrain_types=('p', 'g')):
        """
        Adds a solid castle region of `size` nodes, starting from a random suitable location.
        Only expands into nodes of the given terrain_types (e.g., plains, grassland).
        """
        from collections import deque

        # Find all possible starting points
        candidates = [(y, x) for y in range(self.height) for x in range(self.width)
                    if self.grid[y][x].terrain_type in terrain_types]
        if not candidates:
            print("No suitable starting point for castle.")
            return

        start_y, start_x = random.choice(candidates)
        castle_nodes = set()
        queue = deque()
        queue.append((start_y, start_x))
        castle_nodes.add((start_y, start_x))

        while queue and len(castle_nodes) < size:
            cy, cx = queue.popleft()
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    (ny, nx) not in castle_nodes and
                    self.grid[ny][nx].terrain_type in terrain_types):
                    castle_nodes.add((ny, nx))
                    queue.append((ny, nx))
                    if len(castle_nodes) >= size:
                        break

        # Assign castle terrain type
        for y, x in castle_nodes:
            self.grid[y][x].terrain_type = 'K'
            self.grid[y][x].terrain_label = 'Castle'

        print(f"Castle region created with {len(castle_nodes)} nodes.")

    def add_exits_and_road(self, exit1, road_terrain_types=('p', 'g', 'f', 'F', 'H'), num_side_roads=3):
        """
        Adds two exits: one at exit1 (y, x), and one at the farthest reachable border node.
        Draws a winding road ('W') between them using a randomized A*.
        Adds several short side roads branching from the main road.
        """
        from heapq import heappush, heappop

        # 1. Place the first exit
        y1, x1 = exit1
        self.grid[y1][x1].terrain_type = 'E'
        self.grid[y1][x1].terrain_label = 'Exit'

        # 2. Find all border nodes (excluding corners)
        border_nodes = []
        for x in range(1, self.width-1):
            border_nodes.append((0, x))
            border_nodes.append((self.height-1, x))
        for y in range(1, self.height-1):
            border_nodes.append((y, 0))
            border_nodes.append((y, self.width-1))
        border_nodes = [pos for pos in border_nodes if pos != (y1, x1)]

        # 3. BFS to find all reachable border nodes from exit1
        visited = set()
        queue = deque()
        queue.append((y1, x1))
        visited.add((y1, x1))
        reachable = set()
        while queue:
            cy, cx = queue.popleft()
            if (cy, cx) in border_nodes:
                reachable.add((cy, cx))
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    (ny, nx) not in visited and
                    self.grid[ny][nx].terrain_type in road_terrain_types):
                    visited.add((ny, nx))
                    queue.append((ny, nx))

        # 4. Pick the farthest reachable border node
        if reachable:
            y2, x2 = max(reachable, key=lambda pos: (pos[0]-y1)**2 + (pos[1]-x1)**2)
        else:
            y2, x2 = max(border_nodes, key=lambda pos: (pos[0]-y1)**2 + (pos[1]-x1)**2)
        self.grid[y2][x2].terrain_type = 'E'
        self.grid[y2][x2].terrain_label = 'Exit'

        # 5. Randomized A* for winding main road
        def heuristic(a, b):
            # Add a small random value to the heuristic to encourage winding
            return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5 + random.uniform(0, 1.1)

        open_set = []
        heappush(open_set, (0 + heuristic((y1, x1), (y2, x2)), 0, (y1, x1), []))
        closed = set()
        main_road = set()
        while open_set:
            est_total, cost, (cy, cx), path = heappop(open_set)
            if (cy, cx) == (y2, x2):
                for py, px in path:
                    if self.grid[py][px].terrain_type not in ('E', 'X'):
                        self.grid[py][px].terrain_type = 'W'
                        self.grid[py][px].terrain_label = 'Road'
                        main_road.add((py, px))
                break
            if (cy, cx) in closed:
                continue
            closed.add((cy, cx))
            neighbors = []
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    self.grid[ny][nx].terrain_type in road_terrain_types + ('E',)):
                    neighbors.append((ny, nx))
            # Shuffle neighbors to add more randomness
            random.shuffle(neighbors)
            for ny, nx in neighbors:
                if (ny, nx) not in closed:
                    heappush(open_set, (cost+1+heuristic((ny, nx), (y2, x2)), cost+1, (ny, nx), path+[(ny, nx)]))

        # 6. Add side roads branching from the main road
        main_road_list = list(main_road)
        for _ in range(num_side_roads):
            if not main_road_list:
                break
            start = random.choice(main_road_list)
            length = random.randint(8, 18)
            cy, cx = start
            branch_path = []
            for _ in range(length):
                candidates = []
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and
                        self.grid[ny][nx].terrain_type in road_terrain_types and
                        (ny, nx) not in main_road and (ny, nx) not in branch_path):
                        candidates.append((ny, nx))
                if not candidates:
                    break
                ny, nx = random.choice(candidates)
                self.grid[ny][nx].terrain_type = 'W'
                self.grid[ny][nx].terrain_label = 'Road'
                branch_path.append((ny, nx))
                cy, cx = ny, nx

    def draw_text_map(self):
        """Prints a simple text-based representation of the map to the console."""
        print("\n--- Text Map ---")
        for y in range(self.height):
            row = ''.join([self.grid[y][x].terrain_type for x in range(self.width)])
            print(row)

    def draw_graphical_map(self):
        """Uses matplotlib to draw a color-coded image of the map, with a Save PNG button."""
        print("\nGenerating graphical map (close window to exit)...")
        
        color_map = {
            'D': [26/255, 56/255, 97/255],
            'R': [108/255, 157/255, 202/255],
            'S': [230/255, 216/255, 170/255],
            'g': [152/255, 179/255, 103/255],
            'p': [192/255, 179/255, 115/255],
            'f': [103/255, 146/255, 99/255],
            'F': [54/255, 79/255, 59/255],
            'H': [153/255, 119/255, 84/255],
            'M': [100/255, 100/255, 100/255],
            'C': [45/255, 45/255, 45/255],
            'P': [248/255, 250/255, 255/255],
            'K': [90/255, 110/255, 125/255],  # Castle (blue gray)
            'W': [133/255, 135/255, 126/255],  # Road (brownish)
            'E': [1.0, 0.7, 0.2],  # Exit (gold)
            'X': [0.2, 0.2, 0.2]  # Impassable (gray)
        }

        image_data = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                terrain = self.grid[y][x].terrain_type
                image_data[y, x] = color_map.get(terrain, [0, 0, 0])
    
        # 🪄 Resample the image using smooth zoom, then show with no interpolation
        upscale_factor = 8  # You can tweak this (e.g., 4-8 for 100x100 maps)
        smoothed_image = zoom(image_data, (upscale_factor, upscale_factor, 1), order=1)  # cubic interpolation

        fig, ax = plt.subplots(figsize=(12, 12))
        plt.subplots_adjust(bottom=0.15)
        ax.imshow(smoothed_image, interpolation='none')  # Don't blur further
        ax.axis('off')  # cleaner export
        ax.set_title(f"Procedural World Map (Seed: {self.seed})")

        # Create a legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color_map[key], label=f"{key}: {label}")
                           for key, label in self.get_legend_labels().items()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticks([])
        ax.set_yticks([])
        # Example: plot a little tree icon at (x, y)
        
        tree_img = mpimg.imread('tree_icon.png')  # Use a small PNG with transparency
        peakL_img = mpimg.imread('peakL_icon.png')  # Use a small PNG with transparency
        wave_img = mpimg.imread('wave_icon.png')  # Use a small PNG with transparency
        grass_img = mpimg.imread('grass_icon.png')  # Use a small PNG with transparency
        tower_img = mpimg.imread('tower_icon.png')  # Use a small PNG with transparency

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'F' and random.random() < 0.3:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(tree_img, zoom=0.06)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)
            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'f' and random.random() < 0.1:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(tree_img, zoom=0.06)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)
            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'M' and random.random() < 0.04:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(peakL_img, zoom=0.09)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)

            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'P' and random.random() < 0.06:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(peakL_img, zoom=0.12)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)
            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'D' and random.random() < 0.01:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(wave_img, zoom=0.012)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)

            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'g' and random.random() < 0.2:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(grass_img, zoom=0.01)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)

            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'f' and random.random() < 0.06:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(grass_img, zoom=0.01)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)

            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'p' and random.random() < 0.04:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(grass_img, zoom=0.01)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)
            for x in range(self.width):
                if self.grid[y][x].terrain_type == 'K' and random.random() < 0.04:
                    ux, uy = x * upscale_factor, y * upscale_factor
                    imagebox = OffsetImage(tower_img, zoom=0.035)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
                    ax.add_artist(ab)

        ax.plot([x], [y], marker='^', color='darkgreen', markersize=4)
        # Add Save PNG button
        ax_save = plt.axes([0.4, 0.03, 0.2, 0.06])  # x, y, width, height
        btn_save = Button(ax_save, 'Save as PNG', color='#cccccc', hovercolor='#aaaaaa')

        # 1. Define the callback function for the JSON button
        def save_json(event):
            self.export_to_json() # Calls the export method

        # 2. Define the callback function for the PNG button
        def save_png(event):
            fig.savefig('generated_map.png', bbox_inches='tight', dpi=300)
            print("Map image saved as generated_map.png")

        # 3. Create axes for the buttons, placing them side-by-side
        # [left, bottom, width, height]
        ax_save_png = plt.axes([0.31, 0.03, 0.18, 0.06])
        ax_save_json = plt.axes([0.51, 0.03, 0.18, 0.06])

        # 4. Create the button widgets on their respective axes
        btn_save_png = Button(ax_save_png, 'Save as PNG', color='#cccccc', hovercolor='#aaaaaa')
        btn_save_json = Button(ax_save_json, 'Save as JSON', color='#cccccc', hovercolor='#aaaaaa')

        # 5. Add Import JSON button
        ax_import_json = plt.axes([0.71, 0.03, 0.18, 0.06])
        btn_import_json = Button(ax_import_json, 'Import JSON', color='#cccccc', hovercolor='#aaaaaa')

        def import_json(event):
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(
                title="Select map JSON file",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                self.import_from_json(filename)
                plt.close(fig)
                self.draw_graphical_map()  # Redraw with the imported map

        

        # 6. Connect the buttons to their callback functions
        btn_save_png.on_clicked(save_png)
        btn_save_json.on_clicked(save_json)
        btn_import_json.on_clicked(import_json)
        plt.show()
        
  
    @staticmethod    
    def get_legend_labels():
        """Returns a dictionary of labels for the legend."""
        return {
            'D': 'Deep Water', 'R': 'River', 'S': 'Sandy Beach',
            'g': 'Grassland', 'p': 'Plains', 'f': 'Forested Plains',
            'F': 'Dense Forest', 'H': 'Highlands', 'M': 'Rocky Mountain',
            'C': 'Cliff', 'P': 'Snowy Peak', 'K': 'Castle', 'W': 'Road', 
            'E': 'Exit', 'X': 'Impassable'
        }

    def get_node_details(self, x, y):
        """Returns the full details of a specific node."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return "Coordinates out of bounds."
    
    def export_to_json(self, filename="world_map.json"):
        """Exports the generated map data to a JSON file."""
        map_data = {
            "map_metadata": {
                "width": self.width,
                "height": self.height,
                "seed": self.seed
            },
            "nodes": []
        }

        for y in range(self.height):
            row = []
            for x in range(self.width):
                node = self.grid[y][x]
                # Basic connections (can be enhanced)
                connections = {}
                for dx, dy, name in [(0,-1,"N"), (1,0,"E"), (0,1,"S"), (-1,0,"W"), (1,-1,"NE"), (1,1,"SE"), (-1,1,"SW"), (-1,-1,"NW")]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        connections[name] = (nx, ny)

                node_data = {
                    "coords": {"x": x, "y": y},
                    "terrain": {
                        "type": node.terrain_type,
                        "label": node.terrain_label,
                        "elevation": round(node.elevation, 2),
                    },
                    "description_base": f"A {node.terrain_label.lower()} area.",
                    "connections": connections,
                    "gameplay": {
                        "visited": False,
                        "difficulty_class": 10 # Placeholder DC
                    }
                }
                row.append(node_data)
            map_data["nodes"].append(row)

        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        print(f"Map data successfully exported to {filename}")

    def import_from_json(self, filename):
        """Loads map data from a JSON file and updates the grid."""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.width = data["map_metadata"]["width"]
        self.height = data["map_metadata"]["height"]
        self.seed = data["map_metadata"].get("seed", 0)
        self.grid = [[Node(x, y) for x in range(self.width)] for y in range(self.height)]
        for y, row in enumerate(data["nodes"]):
            for x, node_data in enumerate(row):
                node = self.grid[y][x]
                node.terrain_type = node_data["terrain"]["type"]
                node.terrain_label = node_data["terrain"]["label"]
                node.elevation = node_data["terrain"].get("elevation", 0.0)

    

def verify_world_map_json(filename="world_map.json"):
    import json
    data = json.load(open(filename))
    width = data["map_metadata"]["width"]
    height = data["map_metadata"]["height"]
    nodes = data["nodes"]
    errors = []
    for y, row in enumerate(nodes):
        for x, node in enumerate(row):
            cx, cy = node["coords"]["x"], node["coords"]["y"]
            if cx != x or cy != y:
                errors.append(f"Coord mismatch at ({x},{y}): got ({cx},{cy})")
            ttype = node["terrain"]["type"]
            if ttype not in MapGenerator.get_legend_labels():
                errors.append(f"Unknown terrain type '{ttype}' at ({x},{y})")
            if not node["terrain"]["label"]:
                errors.append(f"Missing label at ({x},{y})")
            for dir, (nx, ny) in node["connections"].items():
                if not (0 <= nx < width and 0 <= ny < height):
                    errors.append(f"Invalid connection {dir} from ({x},{y}) to ({nx},{ny})")
            if "visited" not in node["gameplay"] or "difficulty_class" not in node["gameplay"]:
                errors.append(f"Missing gameplay fields at ({x},{y})")
    if errors:
        print("Validation errors found:")
        for err in errors:
            print(err)
    else:
        print("All nodes in world_map.json are valid.")


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_world_map_json()
    else:
        # Create and generate the map
        generator = MapGenerator(width=100, height=100)
        world_grid = generator.generate_world()
        
        # Add this line to generate the castle region
        generator.add_castle_region(size=100, terrain_types=('p', 'g'))
        # Example: place first exit at (1, 75)
        generator.add_exits_and_road(exit1=(1, 75))

        # --- Display the results ---
        # 1. Simple console output
        generator.draw_text_map()
        
        # 2. Detailed query of a specific node
        print("\n--- Node Details ---")
        node_info = generator.get_node_details(50, 50)
        print(node_info)

        # 3. Graphical map (most informative)
        generator.draw_graphical_map()
