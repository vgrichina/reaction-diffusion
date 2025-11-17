"""
Flask HTTP server for THRML Gray-Scott simulation
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time
from simulation import SimulationSession

app = Flask(__name__, static_folder='static')
CORS(app)

# Global simulation session
session = None
session_lock = threading.Lock()

# Background simulation thread
background_thread = None
simulation_running = False


def background_simulator():
    """Background thread that continuously advances simulation"""
    global session, simulation_running

    while simulation_running:
        with session_lock:
            if session:
                try:
                    session.simulate_steps(1)
                except Exception as e:
                    print(f"Simulation error: {e}")

        # Target ~30 FPS for background simulation
        time.sleep(1.0 / 30.0)


@app.route('/')
def index():
    """Serve the client HTML"""
    return send_from_directory('static', 'index.html')


@app.route('/api/init', methods=['POST'])
def init_simulation():
    """Initialize a new simulation session"""
    global session, simulation_running, background_thread

    data = request.json or {}
    rows = data.get('rows', 256)
    cols = data.get('cols', 256)

    with session_lock:
        session = SimulationSession(rows, cols)

    # Start background simulation thread
    if not simulation_running:
        simulation_running = True
        background_thread = threading.Thread(target=background_simulator, daemon=True)
        background_thread.start()

    return jsonify({
        'status': 'success',
        'rows': rows,
        'cols': cols,
        'maxStep': 0
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get simulation state at specific step"""
    step = request.args.get('step', type=int)

    with session_lock:
        if not session:
            return jsonify({'error': 'No active session'}), 400

        state = session.get_state(step)

    if state is None:
        return jsonify({'error': 'Invalid step'}), 400

    return jsonify(state)


@app.route('/api/params', methods=['POST'])
def update_params():
    """Update simulation parameters"""
    data = request.json

    params = {}
    if 'F' in data:
        params['F'] = float(data['F'])
    if 'k' in data:
        params['k'] = float(data['k'])
    if 'Du' in data:
        params['Du'] = float(data['Du'])
    if 'Dv' in data:
        params['Dv'] = float(data['Dv'])

    with session_lock:
        if not session:
            return jsonify({'error': 'No active session'}), 400

        session.update_params(**params)

    return jsonify({'status': 'success', 'params': params})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to initial state"""
    data = request.json or {}
    pattern = data.get('pattern', 'stripes')

    with session_lock:
        if not session:
            return jsonify({'error': 'No active session'}), 400

        session.reset(pattern)

    return jsonify({'status': 'success'})


@app.route('/api/interact', methods=['POST'])
def interact():
    """Add chemicals at position"""
    data = request.json
    x = int(data['x'])
    y = int(data['y'])
    brush_size = int(data.get('brushSize', 10))

    with session_lock:
        if not session:
            return jsonify({'error': 'No active session'}), 400

        session.interact(x, y, brush_size)

    return jsonify({'status': 'success'})


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get range of historical frames"""
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', type=int)
    stride = request.args.get('stride', 1, type=int)

    with session_lock:
        if not session:
            return jsonify({'error': 'No active session'}), 400

        frames = session.get_history_range(start, end, stride)

    return jsonify({'frames': frames})


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run N simulation steps"""
    data = request.json
    steps = int(data.get('steps', 10))

    with session_lock:
        if not session:
            return jsonify({'error': 'No active session'}), 400

        session.simulate_steps(steps)
        max_step = len(session.history_v) - 1

    return jsonify({'status': 'success', 'maxStep': max_step})


@app.route('/api/pause', methods=['POST'])
def pause_simulation():
    """Pause background simulation"""
    global simulation_running
    simulation_running = False
    return jsonify({'status': 'paused'})


@app.route('/api/resume', methods=['POST'])
def resume_simulation():
    """Resume background simulation"""
    global simulation_running, background_thread

    if not simulation_running:
        simulation_running = True
        background_thread = threading.Thread(target=background_simulator, daemon=True)
        background_thread.start()

    return jsonify({'status': 'running'})


if __name__ == '__main__':
    # Initialize default session
    print("Initializing simulation...")
    session = SimulationSession(256, 256)
    print("Starting server...")

    # Start background simulation
    simulation_running = True
    background_thread = threading.Thread(target=background_simulator, daemon=True)
    background_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
