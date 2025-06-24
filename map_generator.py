import numpy as np
import noise
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from scipy.ndimage import zoom
import random 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

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

    # Add 'K' to your color_map and legend in draw_graphical_map and get_legend_labels:
    # 'K': [160/255, 82/255, 45/255],  # Example: brown for castle

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
            'K': [90/255, 110/255, 125/255],  # Castle (brown)
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

        
            
        # Create a legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color_map[key], label=f"{key}: {label}")
                           for key, label in self.get_legend_labels().items()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(f"Procedural World Map (Seed: {self.seed})")
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

        def save_png(event):
            fig.savefig('generated_map.png', bbox_inches='tight')
            print("Map saved as generated_map.png")

        # Optional: Save directly
        fig.savefig("map_output.png", dpi=300, bbox_inches='tight')
        btn_save.on_clicked(save_png)
        plt.show()
        
  
        
    def get_legend_labels(self):
        """Returns a dictionary of labels for the legend."""
        return {
            'D': 'Deep Water', 'R': 'River', 'S': 'Sandy Beach',
            'g': 'Grassland', 'p': 'Plains', 'f': 'Forested Plains',
            'F': 'Dense Forest', 'H': 'Highlands', 'M': 'Rocky Mountain',
            'C': 'Cliff', 'P': 'Snowy Peak', 'K': 'Castle'
        }

    def get_node_details(self, x, y):
        """Returns the full details of a specific node."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return "Coordinates out of bounds."

# --- Main Execution ---
if __name__ == "__main__":
    # Create and generate the map
    generator = MapGenerator(width=100, height=100)
    world_grid = generator.generate_world()
    
    # Add this line to generate the castle region
    generator.add_castle_region(size=100, terrain_types=('p', 'g'))

    # --- Display the results ---
    # 1. Simple console output
    generator.draw_text_map()
    
    # 2. Detailed query of a specific node
    print("\n--- Node Details ---")
    node_info = generator.get_node_details(50, 50)
    print(node_info)

    # 3. Graphical map (most informative)
    generator.draw_graphical_map()
