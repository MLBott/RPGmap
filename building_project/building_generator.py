import random
from typing import List, Dict, Set, Tuple

def generate_building_interior(nodes: List[Dict], building_type: str, density: str) -> List[Dict]:
    """
    Generates an interior layout by modifying node descriptions within a unified node list.
    
    Args:
        nodes: The list of local node objects for the current floor.
        building_type: The type of building (for future rule expansion).
        density: The desired room density (for future rule expansion).
        
    Returns:
        The updated list of nodes with new descriptions for generated rooms.
    """
    # Create a dictionary for quick lookups of nodes by their coordinates
    grid_map = {(n['coords']['x'], n['coords']['y']): n for n in nodes}
    processed_coords = set()
    
    # Identify all coordinates that are not walls and can become rooms
    interior_coords = {
        (n['coords']['x'], n['coords']['y']) for n in nodes
        if not n.get('description_base', '').lower().startswith('wall.')
    }
    
    room_counter = 1
    
    # Use flood fill to find contiguous areas within the interior
    for x, y in interior_coords:
        if (x, y) not in processed_coords:
            # Find all connected nodes that form a single area
            area_coords = _flood_fill((x, y), interior_coords, processed_coords)
            
            if area_coords:
                # Assign a random room type to all nodes in this area
                # This is where more complex room-fitting logic would go
                room_types = ['Bedroom', 'Dining Room', 'Hallway', 'Kitchen', 'Study', 'Library', 'Armory']
                chosen_room_type = random.choice(room_types)
                room_name = f"{chosen_room_type} {room_counter}"
                
                for ax, ay in area_coords:
                    node_in_area = grid_map.get((ax, ay))
                    if node_in_area:
                        # Update the node's description based on the new room type
                        node_in_area['description_base'] = f"{chosen_room_type}. This is the {room_name.lower()}."
                
                room_counter += 1
                
    # Return the modified list of all nodes
    return list(grid_map.values())

def _flood_fill(start_coord: Tuple[int, int], available_coords: Set[Tuple[int, int]], processed_coords: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    A basic flood fill algorithm to find a connected area of coordinates.
    """
    if start_coord in processed_coords:
        return set()
        
    area = set()
    stack = [start_coord]
    
    while stack:
        x, y = stack.pop()
        
        # Skip if already processed or not a valid interior coordinate
        if (x, y) in processed_coords or (x, y) not in available_coords:
            continue
            
        processed_coords.add((x, y))
        area.add((x, y))
        
        # Add cardinal neighbors to the stack to continue the fill
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))
        
    return area

