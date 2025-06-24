import random
import json
from enum import Enum, auto

# --- Constants ---
NUM_COLS = 7
NUM_ROWS = 15  # Game floors 1-15 correspond to rows 0-14
NUM_PATH_GENERATIONS = 6

# Ascension 20 Location Odds (as per user's initial description)
LOCATION_ODDS_A20 = {
    'MONSTER': 0.45,
    'ELITE': 0.16,
    'MERCHANT': 0.05,
    'REST_SITE': 0.11,
    'TREASURE': 0.03,  # Note: Floor 9 is always Treasure
    'EVENT': 0.20,
}

# --- Enums and Helper Classes ---

class LocationType(Enum):
    EMPTY = auto()      # For nodes not yet assigned or pruned
    MONSTER = auto()
    ELITE = auto()
    MERCHANT = auto()
    REST_SITE = auto()
    TREASURE = auto()
    EVENT = auto()
    BOSS = auto()

    def to_json(self):
        """Helper for JSON serialization of the enum member name."""
        return self.name

class Node:
    """Represents a single room or node on the map."""
    def __init__(self, row, col):
        self.row = row  # 0 to NUM_ROWS - 1
        self.col = col  # 0 to NUM_COLS - 1
        self.id = f"N-{row}-{col}"  # Unique ID for the node
        self.location_type = LocationType.EMPTY
        self.children = []  # List of connected Node objects on the next floor
        self.parents = []   # List of connected Node objects on the previous floor
        self.is_part_of_path = False # Flag to check if node is used in any generated path

    def __repr__(self):
        return f"<Node {self.id} ({self.location_type.name})>"

    def to_json(self):
        """Converts node data to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "type": self.location_type.to_json() if self.location_type else LocationType.EMPTY.to_json(),
            "is_part_of_path": self.is_part_of_path,
            "is_boss": self.location_type == LocationType.BOSS
        }

# --- Core Map Generation Logic ---

class MapGenerator:
    """Generates the Slay the Spire style map."""
    def __init__(self, ascension_level=20):
        self.ascension_level = ascension_level # Could be used for different odds later
        self.grid = {} # Using a dictionary: {(row, col): Node_instance}
        self.paths = [] # Stores lists of nodes, each list being a path
        self.all_edges_coords = set() # Stores ( (r1,c1), (r2,c2) ) tuples for path crossing checks
        self.boss_node_object = None # Stores the actual Boss Node object

        # Initialize the grid with Node objects
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                self.grid[(r, c)] = Node(r, c)

    def _get_node(self, row, col):
        """Safely retrieves a node from the grid."""
        return self.grid.get((row, col))

    def _get_potential_next_nodes(self, current_node_row, current_node_col):
        """
        Gets up to 3 closest rooms on the next floor.
        Connections are typically to (row+1, col-1), (row+1, col), (row+1, col+1).
        """
        next_row = current_node_row + 1
        if next_row >= NUM_ROWS: # Cannot go beyond the last game row
            return []

        potential_nodes = []
        for dc in [-1, 0, 1]: # Delta column
            next_col = current_node_col + dc
            if 0 <= next_col < NUM_COLS: # Check if column is within bounds
                node = self._get_node(next_row, next_col)
                if node:
                    potential_nodes.append(node)
        random.shuffle(potential_nodes) # Randomize selection order
        return potential_nodes

    def _check_path_crossing(self, node1_coords, node2_coords, existing_edges_coords):
        """
        Simplified path crossing check.
        Prevents identical edges and basic visual line crossings on the same floor transition.
        A true geometric check is more complex.
        """
        # Canonical form for the edge (sorted tuple of coordinate tuples)
        new_edge = tuple(sorted((node1_coords, node2_coords)))
        if new_edge in existing_edges_coords:
             return True # This exact edge already exists

        r1, c1 = node1_coords
        r2, c2 = node2_coords

        # This check is primarily for direct forward connections
        if r2 != r1 + 1:
            return False

        # Check for visual crossing with other edges at the same floor transition
        for (e_r1_s, e_c1_s), (e_r2_s, e_c2_s) in existing_edges_coords:
            # Ensure edges are ordered for consistent comparison
            (er1, ec1), (er2, ec2) = tuple(sorted(((e_r1_s, e_c1_s), (e_r2_s, e_c2_s))))

            if er1 == r1 and er2 == r2: # Comparing edges starting and ending on the same floors
                # If new path starts left of existing path, it must end left_or_equal to avoid visual crossing
                if c1 < ec1 and c2 > ec2: return True
                # If new path starts right of existing path, it must end right_or_equal
                if c1 > ec1 and c2 < ec2: return True
        return False


    def _generate_single_path(self, start_node_col, existing_paths_edges_coords):
        """
        Generates one full path from Floor 1 (row 0) up to Floor 15 (row 14).
        Returns the list of nodes in this path and the coordinate pairs of edges created.
        """
        current_path_nodes = []
        current_path_edges_coords = set() # Edges specific to this path being built

        current_node = self._get_node(0, start_node_col)
        if not current_node: return [], set()
        current_path_nodes.append(current_node)

        for _r in range(NUM_ROWS - 1): # Iterate to connect from row 0 up to row 13 (to connect to row 14)
            potential_next_options = self._get_potential_next_nodes(current_node.row, current_node.col)
            if not potential_next_options:
                break # Dead end, path cannot continue

            chosen_next_node = None
            current_node_coords = (current_node.row, current_node.col)

            # Try to find a connection that doesn't cross existing paths (from all_edges_coords + current_path_edges_coords)
            for next_node_candidate in potential_next_options:
                candidate_node_coords = (next_node_candidate.row, next_node_candidate.col)
                # Check against all global edges AND edges already made in this current path
                if not self._check_path_crossing(current_node_coords, candidate_node_coords, existing_paths_edges_coords | current_path_edges_coords):
                    chosen_next_node = next_node_candidate
                    break
            
            if not chosen_next_node and potential_next_options: # Fallback: if all options cross, pick the first one
                chosen_next_node = potential_next_options[0]
            elif not chosen_next_node: # No options at all (should be caught by `if not potential_next_options: break`)
                break # Should not happen if potential_next_options was not empty

            # Add connection
            current_node.children.append(chosen_next_node)
            chosen_next_node.parents.append(current_node)
            
            # Add edge coordinates (canonical form)
            new_edge_tuple = tuple(sorted((current_node_coords, (chosen_next_node.row, chosen_next_node.col))))
            current_path_edges_coords.add(new_edge_tuple)

            current_node = chosen_next_node
            current_path_nodes.append(current_node)

        return current_path_nodes, current_path_edges_coords

    def generate_map_layout(self):
        """
        Generates the paths for the map according to Slay the Spire rules.
        """
        self.paths = []
        self.all_edges_coords = set() # Reset global edges
        starting_nodes_floor1_cols = [] # Track starting columns for first 2 paths

        for i in range(NUM_PATH_GENERATIONS):
            chosen_start_col = -1
            # Rule: First 2 paths must have different start nodes on Floor 1
            attempts = 0 # Safety break for while loop
            while attempts < NUM_COLS * 2:
                potential_start_col = random.randint(0, NUM_COLS - 1)
                if i < 2: # For the first two paths
                    if potential_start_col not in starting_nodes_floor1_cols:
                        chosen_start_col = potential_start_col
                        starting_nodes_floor1_cols.append(chosen_start_col)
                        break
                else: # For subsequent paths, can reuse start nodes
                    chosen_start_col = potential_start_col
                    break
                attempts +=1
            
            if chosen_start_col == -1: # Fallback if unique start not found
                available_cols = [c for c in range(NUM_COLS) if c not in starting_nodes_floor1_cols]
                chosen_start_col = random.choice(available_cols if available_cols else list(range(NUM_COLS)))
                if i < 2 and chosen_start_col not in starting_nodes_floor1_cols:
                     starting_nodes_floor1_cols.append(chosen_start_col)


            path_nodes, path_edges_coords = self._generate_single_path(chosen_start_col, self.all_edges_coords)
            
            if path_nodes:
                self.paths.append(path_nodes)
                self.all_edges_coords.update(path_edges_coords) # Add new edges to global set
                for node in path_nodes:
                    node.is_part_of_path = True # Mark node as active
            # else:
                # print(f"Warning: Path {i+1} could not be fully generated from col {chosen_start_col}.")


    def _get_random_location_type(self):
        """Return a random location type based on A20 odds."""
        rand_val = random.random()
        cumulative = 0
        for loc_type_str, probability in LOCATION_ODDS_A20.items():
            cumulative += probability
            if rand_val < cumulative:
                return LocationType[loc_type_str]
        return LocationType.MONSTER # Fallback, should not be reached if odds sum to 1

    def _check_location_rules(self, node, new_loc_type):
        """
        Checks all assignment rules for a given node and potential new location type.
        Returns True if valid, False otherwise.
        """
        # Rule: Elite and Rest Sites can’t be assigned below the 6th Floor (row 5).
        if new_loc_type in [LocationType.ELITE, LocationType.REST_SITE] and node.row < 5: # Floor 6 is row 5
            return False

        # Rule: Elite, Merchant and Rest Site cannot be consecutive.
        consecutive_restricted_types = [LocationType.ELITE, LocationType.MERCHANT, LocationType.REST_SITE]
        if new_loc_type in consecutive_restricted_types:
            # Check parents
            for parent in node.parents:
                if parent.location_type in consecutive_restricted_types:
                    return False
            # Check already assigned children (less common if assigning row-by-row, but good for robustness)
            for child in node.children:
                if child.location_type in consecutive_restricted_types and child.location_type != LocationType.EMPTY: # Check if child is already assigned
                    return False
        
        # Rule: A Room that has 2 or more Paths going out must have all destinations be unique.
        # This check is for when 'node' is being assigned. If its parent has multiple children,
        # 'node' (the current child being assigned) cannot have the same type as an *already assigned* sibling.
        for parent_node in node.parents:
            if len(parent_node.children) >= 2:
                for sibling_node in parent_node.children:
                    if sibling_node != node and sibling_node.location_type == new_loc_type and new_loc_type != LocationType.EMPTY:
                        # print(f"Rule fail: Sibling conflict for {node.id} ({new_loc_type.name}) due to {parent_node.id}'s child {sibling_node.id} ({sibling_node.location_type.name})")
                        return False

        # Rule: Rest Site cannot be on the 14th Floor (row 13).
        if new_loc_type == LocationType.REST_SITE and node.row == 13: # 14th floor is row 13
            return False

        return True

    def assign_locations(self):
        """
        Assigns LocationTypes to all active nodes in the grid based on rules.
        """
        active_nodes = [node for node in self.grid.values() if node.is_part_of_path]

        # 1. Fixed assignments first
        for node in active_nodes:
            if node.row == 0: # 1st Floor (row 0) -> Monsters
                node.location_type = LocationType.MONSTER
            elif node.row == 8: # 9th Floor (row 8) -> Treasure
                node.location_type = LocationType.TREASURE
            elif node.row == 14: # 15th Floor (row 14) -> Rest Sites
                node.location_type = LocationType.REST_SITE
        
        # 2. Assign remaining nodes
        # Sort by row, then column, for a consistent assignment order. This helps with rule checking.
        unassigned_nodes = [n for n in active_nodes if n.location_type == LocationType.EMPTY]
        unassigned_nodes.sort(key=lambda n: (n.row, n.col))

        for node in unassigned_nodes:
            if node.location_type != LocationType.EMPTY: # Skip if already assigned (e.g. fixed)
                continue

            assigned_successfully = False
            # Try assigning based on weighted odds, checking rules.
            # Give a few attempts to find a rule-compliant type via weighted random choice.
            for _ in range(20): # Number of attempts to get a valid type by odds
                potential_loc_type = self._get_random_location_type()
                if self._check_location_rules(node, potential_loc_type):
                    node.location_type = potential_loc_type
                    assigned_successfully = True
                    break
            
            if not assigned_successfully:
                # If weighted random choice fails after several attempts,
                # try iterating through all possible types to find *any* valid one.
                # This prioritizes rule compliance over exact odds if map is too constrained.
                possible_types_shuffled = list(LOCATION_ODDS_A20.keys())
                random.shuffle(possible_types_shuffled)
                for type_str in possible_types_shuffled:
                    potential_loc_type = LocationType[type_str]
                    if self._check_location_rules(node, potential_loc_type):
                        node.location_type = potential_loc_type
                        assigned_successfully = True
                        break

            if not assigned_successfully:
                # Absolute fallback: If no rule-compliant assignment found, assign MONSTER.
                # This should be rare with well-defined rules and map structure.
                # print(f"Warning: Could not assign valid location to {node.id} after all attempts. Defaulting to MONSTER.")
                node.location_type = LocationType.MONSTER


    def add_boss_node(self):
        """
        Adds a Boss Room at the top, connecting to all active rooms on the 15th Floor (row 14).
        """
        # Boss node is conceptually at a row above the last game row (NUM_ROWS)
        boss_row_visual = NUM_ROWS
        boss_col_visual = NUM_COLS // 2 # Centered
        
        self.boss_node_object = Node(boss_row_visual, boss_col_visual)
        self.boss_node_object.location_type = LocationType.BOSS
        self.boss_node_object.is_part_of_path = True # Mark as active for export
        # The boss node is not added to self.grid to avoid messing with NUM_ROWS logic for game floors

        nodes_on_final_floor = [
            n for n in self.grid.values() if n.row == NUM_ROWS - 1 and n.is_part_of_path
        ]
        for node_on_f15 in nodes_on_final_floor:
            node_on_f15.children.append(self.boss_node_object) # Link from F15 to Boss
            self.boss_node_object.parents.append(node_on_f15)  # Link from Boss to F15 (for completeness)


    def export_map_data_for_canvas(self):
        """
        Exports all active nodes and their connections in a format suitable for JSON.
        This data will be used by the HTML/JavaScript canvas renderer.
        """
        nodes_data = []
        edges_data = [] # List of {"from": node_id1, "to": node_id2}

        # Add all active game nodes
        for node in self.grid.values():
            if node.is_part_of_path:
                nodes_data.append(node.to_json())
                # Add edges originating from this node
                for child_node in node.children:
                    if child_node.is_part_of_path: # Ensure child is also active (boss node will be)
                         edges_data.append({"from": node.id, "to": child_node.id})
        
        # Add boss node data if it exists
        if self.boss_node_object:
            boss_json = self.boss_node_object.to_json()
            # Avoid duplicates if it was somehow added through grid iteration (should not happen with current setup)
            if not any(n['id'] == boss_json['id'] for n in nodes_data):
                 nodes_data.append(boss_json)
            # Edges to the boss node were already added when iterating through its parents' children list.

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "config": {
                "numGameRows": NUM_ROWS, # Actual number of playable rows (0 to NUM_ROWS-1)
                "numCols": NUM_COLS,
                # The row index the boss should appear on visually (one above the last game row)
                "visualBossRow": self.boss_node_object.row if self.boss_node_object else NUM_ROWS 
            }
        }

    def generate_full_map(self):
        """Orchestrates the entire map generation process."""
        self.generate_map_layout()
        self.assign_locations()
        self.add_boss_node()
        return self.export_map_data_for_canvas()

# --- Main Execution ---
if __name__ == "__main__":
    generator = MapGenerator(ascension_level=20)
    map_data_for_canvas = generator.generate_full_map()
    
    # Output the generated map data as a JSON string
    # This JSON can then be copied and pasted into the HTML/JS file.
    print(json.dumps(map_data_for_canvas, indent=2))
