import json
from flask import Flask, request, jsonify, render_template
from building_generator import generate_building_interior
from world_manager import find_building_nodes, extract_building_for_editor, update_world_map_nodes

app = Flask(__name__, template_folder='templates')
WORLD_MAP_FILE = 'world_map.json'

@app.route('/')
def home():
    return render_template('buildgen.html')

@app.route('/api/generate', methods=['POST'])
def api_generate_building():
    """API endpoint to generate the building interior using the new node-based logic."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Invalid request: Must be JSON"}), 400
    
    data = request.get_json()
    
    # The frontend now sends the entire list of nodes
    nodes = data.get('nodes')
    building_type = data.get('buildingType')
    density = data.get('density')

    if None in [nodes, building_type, density]:
        return jsonify({"success": False, "error": "Missing 'nodes', 'buildingType', or 'density' in request"}), 400
    
    # Call the updated generator function
    updated_nodes = generate_building_interior(
        nodes=nodes,
        building_type=building_type,
        density=density
    )
    
    # Return the fully updated list of nodes back to the frontend
    return jsonify({"success": True, "nodes": updated_nodes})

# --- (The rest of your server file, including import/export routes, remains the same) ---

@app.route('/api/import-from-world', methods=['POST'])
def api_import_from_world():
    if not request.is_json: return jsonify({"success": False, "error": "Invalid request"}), 400
    data = request.get_json()
    building_label = data.get('building_label')
    grid_size = data.get('gridSize', 50)
    if not building_label: return jsonify({"success": False, "error": "building_label is required"}), 400
    try:
        with open(WORLD_MAP_FILE, 'r', encoding='utf-8') as f: world_data = json.load(f)
        building_nodes = find_building_nodes(world_data, building_label)
        if not building_nodes: return jsonify({"success": False, "error": f"Building '{building_label}' not found."}), 404
        editor_data = extract_building_for_editor(building_nodes, grid_size)
        return jsonify({"success": True, "building_data": editor_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/export-to-world', methods=['POST'])
def api_export_to_world():
    if not request.is_json: return jsonify({"success": False, "error": "Invalid request"}), 400
    edited_building_data = request.get_json()
    try:
        with open(WORLD_MAP_FILE, 'r', encoding='utf-8') as f: world_data = json.load(f)
        updated_world_data = update_world_map_nodes(world_data, edited_building_data)
        with open(WORLD_MAP_FILE, 'w', encoding='utf-8') as f: json.dump(updated_world_data, f, indent=2)
        return jsonify({"success": True, "message": f"Successfully updated '{edited_building_data.get('name')}' in world_map.json"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/preview-world-update', methods=['POST'])
def api_preview_world_update():
    if not request.is_json:
        return jsonify({"success": False, "error": "Invalid request"}), 400
    edited_building_data = request.get_json()
    try:
        with open(WORLD_MAP_FILE, 'r', encoding='utf-8') as f:
            world_data = json.load(f)
        from world_manager import get_updated_nodes_for_preview
        preview_nodes = get_updated_nodes_for_preview(world_data, edited_building_data)
        return jsonify({"success": True, "preview_nodes": preview_nodes})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Building Constructor server at http://127.0.0.1:5005")
    app.run(host='0.0.0.0', port=5005, debug=True)
