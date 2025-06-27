import json
from typing import List, Dict, Set, Tuple

"""
This module handles the logic for interacting with the world_map.json file.
This version contains corrected logic for handling connections between
interior and exterior castle nodes.
"""

def find_building_nodes(world_data: Dict, building_label: str) -> List[Dict]:
    """Finds all nodes in the world map that are part of a specified building."""
    building_nodes = []
    if 'nodes' not in world_data: return []
    for row in world_data.get('nodes', []):
        for node in row:
            if node.get('terrain', {}).get('label') == building_label:
                building_nodes.append(node)
    print(f"Found {len(building_nodes)} nodes for building '{building_label}'.")
    return building_nodes

def extract_building_for_editor(world_nodes: List[Dict], grid_size: int) -> Dict:
    """Extracts building data from world nodes into a local, editable node structure."""
    if not world_nodes: return {}
    min_x = min(node['coords']['x'] for node in world_nodes)
    min_y = min(node['coords']['y'] for node in world_nodes)
    local_nodes = [{
        "coords": {"x": node['coords']['x'] - min_x, "y": node['coords']['y'] - min_y},
        "description_base": node.get('description_base', 'wall.'),
        "terrain": {"label": "Castle"}
    } for node in world_nodes]
    return {
        "name": "Castle", "gridSize": grid_size,
        "floors": [{"nodes": local_nodes, "floor_number": 0}],
        "currentFloor": 0, "world_map_origin": {"x": min_x, "y": min_y}
    }

def get_updated_nodes_for_preview(world_data: Dict, edited_building_data: Dict) -> List[Dict]:
    """
    Calculates the final state of only the edited nodes for a JSON preview,
    implementing all connection rules with corrected logic.
    """
    origin = edited_building_data.get('world_map_origin')
    if not origin: raise ValueError("Missing 'world_map_origin' for preview.")

    updated_nodes_preview = []
    
    world_node_map = {(node['coords']['x'], node['coords']['y']): node for row in world_data['nodes'] for node in row}
    
    DIRECTIONS = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0), "NW": (-1, -1), "NE": (1, -1), "SW": (-1, 1), "SE": (1, 1)}

    for floor_data in edited_building_data.get('floors', []):
        if not floor_data or 'nodes' not in floor_data: continue
            
        local_node_map = {(n['coords']['x'], n['coords']['y']): n for n in floor_data['nodes']}
        wall_locations = {coords for coords, node in local_node_map.items() if node.get('description_base', '').lower().startswith('wall.')}
        
        # Helper function to determine if a node is "exterior" on the fly
        def is_node_exterior(local_coords_to_check: Tuple[int, int]) -> bool:
            for dx, dy in DIRECTIONS.values():
                neighbor_world_coords = (local_coords_to_check[0] + origin['x'] + dx, local_coords_to_check[1] + origin['y'] + dy)
                # If a neighbor is outside the building...
                if neighbor_world_coords not in local_node_map:
                    # ...and it's a non-castle node in the world map...
                    if neighbor_world_coords in world_node_map and world_node_map[neighbor_world_coords].get('terrain', {}).get('label') != 'Castle':
                        return True # ...then this node is exterior.
            return False

        for local_coords, local_node in local_node_map.items():
            world_x, world_y = local_coords[0] + origin['x'], local_coords[1] + origin['y']
            preview_node = { "coords": {"x": world_x, "y": world_y}, "terrain": {"label": "Castle"} }

            desc = local_node.get('description_base', 'wall.')
            if '.' not in desc: desc += '.'
            if not desc.lower().startswith('wall.'):
                 room_name = local_node.get('room_name', desc.split('.')[0].capitalize())
                 desc = f"{room_name}. An interesting room inside the castle."
            preview_node['description_base'] = desc

            new_connections = {}
            
            for direction, (dx, dy) in DIRECTIONS.items():
                neighbor_local_coords = (local_coords[0] + dx, local_coords[1] + dy)
                
                if neighbor_local_coords in local_node_map:
                    # --- Internal Connection Logic ---
                    # Rule 1: Check if types match (wall-to-wall or room-to-room)
                    is_current_wall = local_coords in wall_locations
                    is_neighbor_wall = neighbor_local_coords in wall_locations
                    if is_current_wall != is_neighbor_wall:
                        continue # Skip if one is a wall and the other is not

                    # Rule 2: Check if exterior status matches (interior-to-interior or exterior-to-exterior)
                    is_current_exterior = is_node_exterior(local_coords)
                    is_neighbor_exterior = is_node_exterior(neighbor_local_coords)
                    if is_current_exterior != is_neighbor_exterior:
                        continue # Skip if one is interior and the other is exterior
                    
                    # If both rules pass, create the connection
                    new_connections[direction] = [neighbor_local_coords[0] + origin['x'], neighbor_local_coords[1] + origin['y']]
                else:
                    # --- External Connection Logic ---
                    neighbor_world_coords = (local_coords[0] + origin['x'] + dx, local_coords[1] + origin['y'] + dy)
                    if neighbor_world_coords in world_node_map:
                        if world_node_map[neighbor_world_coords].get('terrain', {}).get('label') != 'Castle':
                             new_connections[direction] = [neighbor_world_coords[0], neighbor_world_coords[1]]
            
            preview_node['connections'] = new_connections
            updated_nodes_preview.append(preview_node)

    return updated_nodes_preview

def update_world_map_nodes(world_data: Dict, edited_building_data: Dict) -> Dict:
    """Updates the main world_map.json data with the detailed, edited building."""
    updated_nodes = get_updated_nodes_for_preview(world_data, edited_building_data)
    world_node_map = {(node['coords']['x'], node['coords']['y']): node for row in world_data['nodes'] for node in row}
    for updated_node in updated_nodes:
        world_coords = (updated_node['coords']['x'], updated_node['coords']['y'])
        if world_coords in world_node_map:
            world_node_map[world_coords].update(updated_node)
            print(f"Updating node ({world_coords[0]}, {world_coords[1]}) in world_map.")
    return world_data
