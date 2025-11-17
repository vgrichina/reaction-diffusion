"""
Dual Simulation Server for Parallel History Comparison
Runs both Native JS-style and THRML simulations side-by-side
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time
import numpy as np
from simulation import SimulationSession

app = Flask(__name__, static_folder='../comparison')
CORS(app)

# Dual simulation state
thrml_session = None
native_history = []  # Stores V grids from native simulation
session_lock = threading.Lock()

# Simulation parameters
ROWS = 256
COLS = 256
TARGET_STEPS = 2000

# Native simulation arrays (server-side implementation)
U_native = None
V_native = None
U_next = None
V_next = None

# Simulation status
simulation_status = {
    'thrml_step': 0,
    'native_step': 0,
    'target': TARGET_STEPS,
    'running': False,
    'complete': False
}

# Simulation parameters
params = {
    'Du': 0.16,
    'Dv': 0.08,
    'F': 0.055,
    'k': 0.062,
    'dt': 1.0
}


def init_native_simulation():
    """Initialize native simulation with same initial conditions as THRML"""
    global U_native, V_native, U_next, V_next, native_history

    U_native = np.ones((ROWS, COLS), dtype=np.float32)
    V_native = np.zeros((ROWS, COLS), dtype=np.float32)
    U_next = np.zeros((ROWS, COLS), dtype=np.float32)
    V_next = np.zeros((ROWS, COLS), dtype=np.float32)

    # Same initialization as THRML: stripes pattern
    for x in range(0, COLS, 20):
        U_native[:, x:x+3] = 0.0
        V_native[:, x:x+3] = 1.0

    # Add same random noise as THRML
    np.random.seed(42)
    U_native += np.random.uniform(-0.01, 0.01, U_native.shape)
    V_native += np.random.uniform(-0.01, 0.01, V_native.shape)

    U_native = np.clip(U_native, 0, 1)
    V_native = np.clip(V_native, 0, 1)

    # Store initial state
    native_history = [V_native.copy()]


def laplacian_native(arr):
    """Compute Laplacian with periodic boundary conditions"""
    return (
        np.roll(arr, 1, axis=0) +   # North
        np.roll(arr, -1, axis=0) +   # South
        np.roll(arr, 1, axis=1) +    # West
        np.roll(arr, -1, axis=1) -   # East
        4 * arr
    )


def step_native():
    """Perform one simulation step using Gray-Scott equations"""
    global U_native, V_native, U_next, V_next, native_history

    lap_U = laplacian_native(U_native)
    lap_V = laplacian_native(V_native)

    uvv = U_native * V_native * V_native

    U_next = U_native + params['dt'] * (
        params['Du'] * lap_U - uvv + params['F'] * (1 - U_native)
    )
    V_next = V_native + params['dt'] * (
        params['Dv'] * lap_V + uvv - (params['F'] + params['k']) * V_native
    )

    U_native = np.clip(U_next, 0, 1).astype(np.float32)
    V_native = np.clip(V_next, 0, 1).astype(np.float32)

    # Store in history
    native_history.append(V_native.copy())


def run_batch_simulation(n_steps):
    """Run both simulations for n_steps"""
    global thrml_session, simulation_status

    print(f"\n{'='*60}")
    print(f"Running batch simulation: {n_steps} steps")
    print(f"{'='*60}\n")

    batch_size = 10  # Run in small batches for better progress tracking
    batches = (n_steps + batch_size - 1) // batch_size

    for batch in range(batches):
        batch_start = time.time()
        steps_this_batch = min(batch_size, n_steps - batch * batch_size)

        # Run native simulation
        for _ in range(steps_this_batch):
            step_native()
            simulation_status['native_step'] = len(native_history) - 1

        # Run THRML simulation
        with session_lock:
            thrml_session.simulate_steps(steps_this_batch)
            simulation_status['thrml_step'] = len(thrml_session.history_v) - 1

        batch_time = time.time() - batch_start

        # Progress update
        total_done = (batch + 1) * batch_size
        progress = min(100, (total_done / n_steps) * 100)

        print(f"Batch {batch+1}/{batches}: "
              f"Native={simulation_status['native_step']}, "
              f"THRML={simulation_status['thrml_step']} "
              f"({batch_time:.2f}s, {progress:.1f}%)")

    print(f"\n{'='*60}")
    print(f"Batch complete!")
    print(f"  Native: {simulation_status['native_step']} steps")
    print(f"  THRML:  {simulation_status['thrml_step']} steps")
    print(f"{'='*60}\n")


def auto_run_to_target():
    """Automatically run both simulations to TARGET_STEPS"""
    global simulation_status

    simulation_status['running'] = True

    start_time = time.time()

    print(f"\n🚀 Starting dual simulation run to {TARGET_STEPS} steps...")

    run_batch_simulation(TARGET_STEPS)

    elapsed = time.time() - start_time

    simulation_status['running'] = False
    simulation_status['complete'] = True

    print(f"\n✅ DUAL SIMULATION COMPLETE!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Native steps: {simulation_status['native_step']}")
    print(f"   THRML steps: {simulation_status['thrml_step']}")
    print(f"   Average: {elapsed/TARGET_STEPS:.3f}s per step")
    print(f"\n🎬 Ready for parallel history comparison!")
    print(f"   URL: http://localhost:5002/parallel-history.html\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the parallel history comparison page"""
    return send_from_directory('../comparison', 'parallel-history.html')


@app.route('/parallel-history.html')
def parallel_history():
    """Serve the parallel history comparison page"""
    return send_from_directory('../comparison', 'parallel-history.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current simulation status"""
    with session_lock:
        return jsonify({
            'thrml_step': simulation_status['thrml_step'],
            'native_step': simulation_status['native_step'],
            'target': simulation_status['target'],
            'running': simulation_status['running'],
            'complete': simulation_status['complete'],
            'thrml_max': len(thrml_session.history_v) - 1 if thrml_session else 0,
            'native_max': len(native_history) - 1
        })


@app.route('/api/thrml/state', methods=['GET'])
def get_thrml_state():
    """Get THRML simulation state at specific step"""
    step = request.args.get('step', type=int)

    with session_lock:
        if not thrml_session:
            return jsonify({'error': 'No THRML session'}), 400

        state = thrml_session.get_state(step)

    if state is None:
        return jsonify({'error': 'Invalid step'}), 400

    return jsonify(state)


@app.route('/api/native/state', methods=['GET'])
def get_native_state():
    """Get native simulation state at specific step"""
    step = request.args.get('step', type=int)

    if step is None:
        step = len(native_history) - 1

    if step < 0 or step >= len(native_history):
        return jsonify({'error': 'Invalid step'}), 400

    v_grid = native_history[step]

    return jsonify({
        'step': step,
        'maxStep': len(native_history) - 1,
        'v_grid': v_grid.flatten().tolist(),
        'rows': ROWS,
        'cols': COLS
    })


@app.route('/api/run', methods=['POST'])
def run_simulation():
    """Run both simulations for N steps"""
    data = request.json or {}
    steps = data.get('steps', 100)

    if simulation_status['running']:
        return jsonify({'error': 'Simulation already running'}), 400

    # Run in background thread
    thread = threading.Thread(target=run_batch_simulation, args=(steps,))
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'steps': steps
    })


@app.route('/api/run-to-target', methods=['POST'])
def run_to_target():
    """Run both simulations to TARGET_STEPS"""
    if simulation_status['running']:
        return jsonify({'error': 'Simulation already running'}), 400

    if simulation_status['complete']:
        return jsonify({
            'status': 'already_complete',
            'thrml_step': simulation_status['thrml_step'],
            'native_step': simulation_status['native_step']
        })

    # Run in background thread
    thread = threading.Thread(target=auto_run_to_target)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'target': TARGET_STEPS
    })


@app.route('/api/reset', methods=['POST'])
def reset_both():
    """Reset both simulations"""
    global thrml_session, simulation_status

    if simulation_status['running']:
        return jsonify({'error': 'Cannot reset while running'}), 400

    with session_lock:
        # Reset THRML
        thrml_session = SimulationSession(ROWS, COLS)

        # Reset native
        init_native_simulation()

        # Reset status
        simulation_status = {
            'thrml_step': 0,
            'native_step': 0,
            'target': TARGET_STEPS,
            'running': False,
            'complete': False
        }

    return jsonify({'status': 'reset_complete'})


@app.route('/api/params', methods=['POST'])
def update_params():
    """Update simulation parameters"""
    global params

    data = request.json

    if 'F' in data:
        params['F'] = float(data['F'])
    if 'k' in data:
        params['k'] = float(data['k'])
    if 'Du' in data:
        params['Du'] = float(data['Du'])
    if 'Dv' in data:
        params['Dv'] = float(data['Dv'])

    # Update THRML session
    with session_lock:
        if thrml_session:
            thrml_session.update_params(**params)

    return jsonify({'status': 'success', 'params': params})


# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DUAL SIMULATION SERVER")
    print("Gray-Scott Reaction-Diffusion: THRML vs Native")
    print("="*60)
    print()

    # Initialize both simulations
    print("Initializing THRML simulation...")
    thrml_session = SimulationSession(ROWS, COLS)

    print("Initializing Native simulation...")
    init_native_simulation()

    print()
    print("✅ Both simulations initialized")
    print()
    print("Server will run at: http://localhost:5002")
    print()
    print("API Endpoints:")
    print("  GET  /api/status           - Get simulation status")
    print("  GET  /api/thrml/state?step=N - Get THRML state")
    print("  GET  /api/native/state?step=N - Get native state")
    print("  POST /api/run              - Run N steps")
    print("  POST /api/run-to-target    - Run to 2000 steps")
    print("  POST /api/reset            - Reset both")
    print()
    print("To run 2000 steps automatically:")
    print("  curl -X POST http://localhost:5002/api/run-to-target")
    print()
    print("="*60)
    print()

    # Auto-start simulation to target
    print("🚀 Auto-starting simulation to 2000 steps...")
    thread = threading.Thread(target=auto_run_to_target)
    thread.daemon = True
    thread.start()

    # Start server
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
