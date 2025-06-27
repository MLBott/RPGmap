from flask import Flask, request, jsonify, render_template
import json
import os
import time
import threading
import google.generativeai as genai
import openai
from collections import deque
from dotenv import load_dotenv
import concurrent.futures
import tempfile
import shutil
from typing import Dict, Set, Tuple, Optional

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)
WORLD_MAP_FILE = 'world_map.json'
# The master prompt template. Placeholders will be filled in by the build_detailed_prompt function.
PROMPT_TEMPLATE = (
    "You are a creative, detail-oriented world-building assistant for a fantasy RPG.\n"
    "Generate an immersive, third-person description for a map node as if the environment is being closely examined.\n\n"
    "## STYLE & TONE\n"
    "- **Perspective:** Describe the node from a close-to-the-ground, spatially immediate perspective. Let the environment speak for itself through texture, orientation, and subtle cues.\n"
    "- **Narration:** Avoid first-person ('I see...') or second-person ('You see...') language entirely. The tone should be objective and descriptive.\n"
    "- **Time:** The description must be **time-agnostic**. DO NOT mention the time of day, sun, moon, or specific weather. Describe only permanent features.\n"
    "- **Output Format:** Do not wrap the output in markdown or JSON. Return only the plain text description.\n\n"
    "## CONTENT INSTRUCTIONS\n"
    "- **Primary Source:** Use the TARGET NODE JSON as the primary source of information for the immediate area.\n"
    "- **Novelty:** Add some unique or novel detail to the core target node to make it memorable.\n"
    "- **Context:** Use the surrounding rings of neighbors for environmental context (e.g., if a forest is next door, mention the sounds of birds or the scent of pine).\n"
    "- **Ring 3 Awareness:** The outermost ring (Ring 3) should only influence the description if it contains large, unmissable features like mountains on the horizon.\n"
    "- **Character Spotlights (Optional):** If it feels natural, optionally include one or two 'character spotlights'—small, interesting details that a character might notice. For example: 'a curiously shaped mushroom', 'a patch of loose scree', or 'an old, moss-covered stone marker'.\n\n"
    "- **Building Nodes** Castle, etc, nodes are three main types: exterior wall, interior wall, and room spaces. Exterior Walls always impassable to interior nodes, description is always from pov outside that incorporates env of adjacent outside non-castle/building nodes."
    "---"
    "## TARGET NODE JSON\n"
    "{target_node_json}\n\n"
    "## CONTEXT: SURROUNDING NEIGHBORS\n\n"
    "### Ring 1 (Full Detail):\n"
    "{ring1_context}\n"
    "\n### Ring 2 (Summary):\n"
    "{ring2_context}\n"
    "\n### Ring 3 (Brief):\n"
    "{ring3_context}\n"
    "\n---\n## DESCRIPTION:\n"
)

# --- GEMINI API SETUP ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_model = None

if API_KEY:
    print("Found GOOGLE_API_KEY from .env file.")
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"\n\n!!! ERROR: Could not configure Gemini API. Error: {e}\n\n")
else:
    print("\n\n!!! WARNING: GOOGLE_API_KEY not found in .env file or environment.")
    print("!!! The crawler cannot run without a configured API Key.")
    print("!!! Please ensure you have a .env file with GOOGLE_API_KEY=YOUR_KEY in it.\n\n")

# --- OPENAI API SETUP ---
openai_client = None
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("Found OPENAI_API_KEY.")
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        openai_client.models.list()
        print("OpenAI API configured successfully.")
    except Exception as e:
        openai_client = None
        print(f"!!! ERROR: Could not configure OpenAI API. Error: {e}")
else:
    print("!!! WARNING: OPENAI_API_KEY not found. OpenAI will not be available.")

# --- CRAWLER STATE MANAGEMENT ---
crawler_state = {
    "is_running": False,
    "todo_list": set(),
    "processing_queue": deque(),
    "completed_nodes": 0,
    "total_nodes": 0,
    "active_workers": set(),
    "thread_object": None,
    "provider": "gemini",
    "executor": None
}

# Separate locks for different operations to prevent deadlocks
file_lock = threading.RLock()  # For file I/O operations
state_lock = threading.RLock()  # For crawler state operations
crawler_lock = threading.Lock()

# --- THREAD-SAFE FILE OPERATIONS ---

def load_map_data():
    """Loads the world map from the JSON file with proper error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with file_lock:
                with open(WORLD_MAP_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        raise ValueError("Empty JSON file")
                    return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON decode error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1)  # Brief pause before retry
                continue
            else:
                raise Exception(f"Failed to load map data after {max_retries} attempts: {e}")
        except Exception as e:
            print(f"Unexpected error loading map data: {e}")
            raise

def save_map_data(data):
    """Saves the world map back to the JSON file with atomic write."""
    try:
        with file_lock:
            # Write to a temporary file first, then rename (atomic operation)
            temp_file = WORLD_MAP_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            if os.path.exists(WORLD_MAP_FILE):
                backup_file = WORLD_MAP_FILE + '.backup'
                shutil.copy2(WORLD_MAP_FILE, backup_file)
            
            shutil.move(temp_file, WORLD_MAP_FILE)
            
    except Exception as e:
        print(f"Error saving map data: {e}")
        # Clean up temp file if it exists
        temp_file = WORLD_MAP_FILE + '.tmp'
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

# --- HELPER FUNCTIONS ---

def is_node_default(node):
    """Checks if a node has the default, un-generated description."""
    return node["description_base"].strip().lower().startswith("a ") and \
           node["description_base"].strip().lower().endswith(" area.")

def build_detailed_prompt(prompt_template, node_context):
    """Builds the comprehensive prompt by filling in a template."""
    target = node_context['target_node']
    
    ring1_text = "\n".join([f"- ({n['coords']['x']},{n['coords']['y']}): {n['description']}" for n in node_context['ring1']])
    ring2_text = "\n".join([f"- ({n['coords']['x']},{n['coords']['y']}): {n.get('summary', 'N/A')}" for n in node_context['ring2']])
    ring3_text = "\n".join([f"- ({n['coords']['x']},{n['coords']['y']}): {n.get('label', 'N/A')}" for n in node_context['ring3']])

    return prompt_template.format(
        target_node_json=json.dumps(target, indent=2),
        ring1_context=ring1_text,
        ring2_context=ring2_text,
        ring3_context=ring3_text
    )

def get_node_context_data(x, y, map_data):
    """Gathers context for a node."""
    nodes = map_data['nodes']
    width = map_data['map_metadata']['width']
    height = map_data['map_metadata']['height']
    context = {"target_node": nodes[y][x], "ring1": [], "ring2": [], "ring3": []}
    
    for ring_dist in range(1, 4):
        for dy in range(-ring_dist, ring_dist + 1):
            for dx in range(-ring_dist, ring_dist + 1):
                if max(abs(dx), abs(dy)) != ring_dist:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = nodes[ny][nx]
                    if ring_dist == 1:
                        context["ring1"].append({
                            "coords": neighbor["coords"], 
                            "description": neighbor["description_base"]
                        })
                    elif ring_dist == 2:
                        context["ring2"].append({
                            "coords": neighbor["coords"], 
                            "summary": f"A {neighbor['terrain']['label']} area."
                        })
                    elif ring_dist == 3:
                        context["ring3"].append({
                            "coords": neighbor["coords"], 
                            "label": neighbor['terrain']['label']
                        })
    return context

def generate_description_with_llm(prompt, provider):
    """Generates a description using the selected LLM provider."""
    if provider == 'openai' and openai_client:
        print("Generating with OpenAI...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a creative, detail-oriented world-building assistant for a fantasy RPG."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
        
    elif provider == 'gemini' and gemini_model:
        print("Generating with Gemini...")
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    else:
        raise ValueError(f"Provider '{provider}' is not available or configured.")

# --- WORKER THREAD FUNCTION ---

def process_node_worker(x, y, prompt_template):
    """
    This is the function that each worker thread will execute.
    It processes a single node and RETURNS its neighbors. It no longer modifies global state.
    """
    node_id = (x, y)
    print(f"Worker: Starting node {node_id}...")
    try:
        # Load data needed for the prompt
        map_data = load_map_data()
        context = get_node_context_data(x, y, map_data)
        prompt = build_detailed_prompt(prompt_template, context)
        
        # --- Make the API call ---
        with crawler_lock:
            # We still need a lock around the API call if the client object is not thread-safe
            selected_provider = crawler_state.get("provider", "gemini")
        new_description = generate_description_with_llm(prompt, selected_provider)
        
        # --- Save the result ---
        map_data = load_map_data()
        map_data['nodes'][y][x]['description_base'] = new_description
        save_map_data(map_data)
        
        print(f"Worker: Finished node {node_id}.")
        
        # Return the original node_id and a list of its valid neighbors
        valid_neighbors = [
            (n['coords']['x'], n['coords']['y']) for n in context.get('ring1', [])
        ]
        return node_id, valid_neighbors

    except Exception as e:
        print(f"Worker: ERROR on node {node_id}: {e}")
        return node_id, [] # Return empty list on failure

# --- CRAWLER MANAGER ---

# In app.py, replace this function:

def crawler_manager(prompt_template):
    """
    The main control loop that now centrally manages the queue and work distribution.
    """
    with crawler_lock:
        print("Manager: Initializing...")
        map_data = load_map_data()
        crawler_state["todo_list"] = {
            (x, y) for y, row in enumerate(map_data['nodes'])
            for x, node in enumerate(row) if is_node_default(node)
        }
        crawler_state["total_nodes"] = len(map_data['nodes']) * len(map_data['nodes'][0])
        crawler_state["completed_nodes"] = crawler_state["total_nodes"] - len(crawler_state["todo_list"])
        print("Manager: Initialization complete.")

    MAX_WORKERS = 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        active_futures = set()

        while True:
            with crawler_lock:
                # Check for stop signal
                if not crawler_state["is_running"]:
                    print("Manager: Stop signal received. Shutting down...")
                    # Attempt to cancel futures that haven't started
                    for f in active_futures:
                        f.cancel()
                    executor.shutdown(wait=True)
                    break

                # Submit new jobs from the queue if there's capacity
                while len(active_futures) < MAX_WORKERS:
                    node_to_process = None
                    if crawler_state["processing_queue"]:
                        node_to_process = crawler_state["processing_queue"].popleft()
                    elif crawler_state["todo_list"]:
                        node_to_process = crawler_state["todo_list"].pop()
                        print(f"Manager: Starting new island at {node_to_process}")
                    
                    while len(active_futures) < MAX_WORKERS:
                        node_to_process = None
                        if crawler_state["processing_queue"]:
                            node_to_process = crawler_state["processing_queue"].popleft()
                        elif crawler_state["todo_list"]:
                            node_to_process = next(iter(crawler_state["todo_list"]))  # Peek, don't pop yet
                            print(f"Manager: Starting new island at {node_to_process}")

                        if node_to_process:
                            # Ensure we don't re-process something
                            if node_to_process in crawler_state["todo_list"]:
                                future = executor.submit(process_node_worker, node_to_process[0], node_to_process[1], prompt_template)
                                active_futures.add(future)
                                crawler_state["active_workers"].add(node_to_process)
                                crawler_state["todo_list"].discard(node_to_process)  # Remove after submitting
                            else:
                                continue # Skip if it was already processed
                        else:
                            break # No more work to submit right now

            # If there's no work left to start and no active workers, we are done.
            if not active_futures and not crawler_state["todo_list"]:
                print("Manager: All work completed. Shutting down.")
                with crawler_lock:
                    crawler_state["is_running"] = False
                break

            # Wait for the next future to complete.
            # Use a timeout to allow the main loop to check the is_running flag periodically.
            try:
                for future in concurrent.futures.as_completed(active_futures, timeout=1.0):
                    completed_node, neighbors = future.result()
                    
                    # This future is done, remove it from the active set
                    active_futures.remove(future)

                    with crawler_lock:
                        crawler_state["completed_nodes"] += 1
                        crawler_state["active_workers"].discard(completed_node)
                        
                        # Add valid, unprocessed neighbors to the queue
                        for neighbor_id in neighbors:
                            if neighbor_id in crawler_state["todo_list"]:
                                crawler_state["processing_queue"].append(neighbor_id)
                                # By removing it here, we ensure it's not picked up by the "island hopper"
                                crawler_state["todo_list"].remove(neighbor_id)
                    break # Process one result per loop iteration
            except concurrent.futures.TimeoutError:
                # No futures completed in the last second, just loop again to check is_running
                continue

# --- FLASK API ENDPOINTS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/map', methods=['GET'])
def get_map():
    try:
        map_data = load_map_data()
        return jsonify(map_data)
    except Exception as e:
        return jsonify({"error": f"Failed to load map: {str(e)}"}), 500

@app.route('/api/node_context', methods=['GET'])
def get_node_context_endpoint():
    try:
        x = int(request.args.get('x'))
        y = int(request.args.get('y'))
        map_data = load_map_data()
        context = get_node_context_data(x, y, map_data)
        return jsonify(context)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_node', methods=['POST'])
def update_node():
    try:
        data = request.json
        map_data = load_map_data()
        map_data['nodes'][data['y']][data['x']]['description_base'] = data['description']
        save_map_data(map_data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- CRAWLER CONTROL ENDPOINTS ---

@app.route('/api/crawler/start', methods=['POST'])
def start_crawler():
    data = request.json
    provider = data.get('provider', 'gemini')
    # Get the user-edited prompt from the request, fall back to the master template
    prompt_template = data.get('prompt', PROMPT_TEMPLATE)

    # (Provider check is unchanged)
    # ...

    with crawler_lock:
        if crawler_state["is_running"]:
            return jsonify({"status": "Already running."})

        print(f"API: Received start command for provider: {provider}")
        crawler_state["provider"] = provider
        crawler_state["is_running"] = True
        
        if not crawler_state["thread_object"] or not crawler_state["thread_object"].is_alive():
            # Pass the prompt_template to the manager thread
            crawler_state["thread_object"] = threading.Thread(target=crawler_manager, args=(prompt_template,), daemon=True)
            crawler_state["thread_object"].start()

        return jsonify({"status": "Crawler start command received."})

@app.route('/api/crawler/stop', methods=['POST'])
def stop_crawler():
    try:
        with state_lock:
            if not crawler_state["is_running"]:
                return jsonify({"status": "Already stopped."})
            
            print("API: Stopping crawler...")
            crawler_state["is_running"] = False
            
        return jsonify({"status": "Crawler stopping..."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/crawler/status', methods=['GET'])
def get_crawler_status():
    try:
        with state_lock:
            status = {
                "is_running": crawler_state["is_running"],
                "completed_nodes": crawler_state["completed_nodes"],
                "total_nodes": crawler_state["total_nodes"],
                "active_workers": list(crawler_state["active_workers"]),
                "queue_size": len(crawler_state["processing_queue"]),
                "todo_count": len(crawler_state["todo_list"])
            }
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/prompt', methods=['GET'])
def get_prompt_template():
    """Endpoint to provide the master prompt template to the frontend."""
    return jsonify({"prompt_template": PROMPT_TEMPLATE})

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)