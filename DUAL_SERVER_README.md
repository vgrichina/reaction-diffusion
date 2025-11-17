# Dual Simulation Server for Parallel History Comparison

## Overview

The dual server runs **both** Native JS-style and THRML simulations server-side, allowing perfect synchronized comparison with unlimited history for both implementations.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Dual Simulation Server (Port 5002)          │
│  ───────────────────────────────────────────────────    │
│                                                          │
│  ┌──────────────────┐       ┌──────────────────┐       │
│  │  Native Sim      │       │  THRML Sim       │       │
│  │  ──────────      │       │  ──────────      │       │
│  │  • NumPy arrays  │       │  • JAX/THRML     │       │
│  │  • Gray-Scott    │       │  • Factor graph  │       │
│  │  • Full history  │       │  • Full history  │       │
│  └──────────────────┘       └──────────────────┘       │
│           │                          │                  │
│           └──────────┬───────────────┘                  │
│                      │                                  │
│              Synchronized Access                        │
│                      │                                  │
└──────────────────────┼──────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Parallel History UI │
            │  • Unified timeline  │
            │  • Side-by-side view │
            │  • Time-travel       │
            └──────────────────────┘
```

## Key Features

1. **Server-Side Both Implementations**
   - Native simulation runs in NumPy (server-side)
   - THRML simulation runs with JAX
   - Both have unlimited history

2. **Auto-Run to 2000 Steps**
   - Server automatically runs both simulations to 2000 steps on startup
   - Progress updates in terminal
   - Ready for comparison when complete

3. **Synchronized Access**
   - Single API to access both histories
   - Same step number retrieves same simulation state
   - Perfect frame-by-frame comparison

4. **Unified Timeline**
   - Scrub through history of both simultaneously
   - Play/pause/step controls
   - Variable playback speed

## Quick Start

```bash
cd thrml-server
python dual_server.py
```

The server will:
1. Initialize both simulations (same initial conditions)
2. **Automatically run 2000 steps** for both
3. Serve the comparison UI at http://localhost:5002

## API Endpoints

### Status
```
GET /api/status
```
Returns current simulation state:
```json
{
  "thrml_step": 2000,
  "native_step": 2000,
  "target": 2000,
  "running": false,
  "complete": true,
  "thrml_max": 2000,
  "native_max": 2000
}
```

### Get THRML State
```
GET /api/thrml/state?step=N
```
Returns THRML simulation at step N.

### Get Native State
```
GET /api/native/state?step=N
```
Returns Native simulation at step N.

### Run Simulations
```
POST /api/run
Body: { "steps": 100 }
```
Run both simulations for N more steps.

### Run to Target
```
POST /api/run-to-target
```
Run both simulations to 2000 steps (if not already complete).

### Reset Both
```
POST /api/reset
```
Reset both simulations to initial state.

### Update Parameters
```
POST /api/params
Body: { "F": 0.055, "k": 0.062 }
```
Update Gray-Scott parameters for both simulations.

## Comparison UI

Open http://localhost:5002/parallel-history.html

### Features

- **Side-by-side canvases**: Native (left) vs THRML (right)
- **Unified timeline**: Single slider controls both
- **Playback controls**:
  - ◄ Step backward
  - ◄◄ Play backward
  - ▶ Play/Pause
  - ▶▶ Play forward
  - ► Step forward
- **Speed control**: 0.25x, 1x, 2x, 4x
- **Pattern presets**: Spots, Stripes, Spirals, Worms
- **Actions**:
  - Run 2000 Steps (if not complete)
  - Reset Both
  - Jump to Start/End

## Implementation Details

### Native Simulation (Server-Side)

```python
def step_native():
    """Gray-Scott equations in NumPy"""
    lap_U = laplacian_native(U_native)  # Periodic boundaries
    lap_V = laplacian_native(V_native)

    uvv = U_native * V_native * V_native

    U_next = U_native + dt * (Du * lap_U - uvv + F * (1 - U_native))
    V_next = V_native + dt * (Dv * lap_V + uvv - (F + k) * V_native)

    # Store in history
    native_history.append(V_native.copy())
```

### THRML Simulation

Uses the existing `SimulationSession` class from `simulation.py` with full THRML factor graph.

### Synchronization

Both simulations:
- Use **same random seed** (42)
- Same **initial pattern** (stripes)
- Same **parameters** (F, k, Du, Dv, dt)
- Run in **lockstep** for comparison

## Performance

Typical run on 256x256 grid:

- **Native**: ~0.01s per step (NumPy, CPU)
- **THRML**: ~0.5s per step (JAX compilation + compute)
- **2000 steps**: ~17 minutes total

Progress is displayed in terminal:

```
Batch 1/200: Native=10, THRML=10 (0.52s, 5.0%)
Batch 2/200: Native=20, THRML=20 (0.51s, 10.0%)
...
✅ DUAL SIMULATION COMPLETE!
   Total time: 1020.3s (17.0 min)
   Native steps: 2000
   THRML steps: 2000
```

## Comparison with Original Design

### Original (Two Separate Systems)
- Native: Client-side JavaScript, 1000 frame ring buffer
- THRML: Server-side Python, unlimited history
- Challenge: Different implementations, hard to sync

### New Dual Server
- Native: Server-side NumPy, unlimited history
- THRML: Server-side JAX/THRML, unlimited history
- Advantage: Perfect synchronization, same exact simulation

## Use Cases

1. **Algorithm Validation**
   - Verify both implementations produce same results
   - Frame-by-frame comparison

2. **Performance Analysis**
   - Compare computation time
   - Native (NumPy) vs THRML (JAX)

3. **Educational**
   - Show "simple" vs "factor graph" approach
   - Demonstrate time-reversible simulations

4. **Demo/Presentation**
   - Beautiful side-by-side visualization
   - Interactive exploration

## Troubleshooting

**Server doesn't start?**
```bash
# Check dependencies
pip install flask flask-cors numpy jax equinox networkx

# Check THRML is installed
cd ../../thrml
pip install -e .
```

**Simulations diverge?**
- Both use same seed and initial conditions
- Should be identical (within floating point precision)
- Check parameters are synchronized

**Slow performance?**
- THRML step time dominated by JAX compilation + sampling
- Native is much faster (plain NumPy)
- Consider reducing target steps or grid size

## Files

- `dual_server.py` - Main dual simulation server
- `simulation.py` - THRML simulation implementation
- `../comparison/parallel-history.html` - Comparison UI

## Example Session

```bash
$ cd thrml-server
$ python dual_server.py

============================================================
DUAL SIMULATION SERVER
Gray-Scott Reaction-Diffusion: THRML vs Native
============================================================

Initializing THRML simulation...
Initializing Native simulation...

✅ Both simulations initialized

Server will run at: http://localhost:5002

🚀 Auto-starting simulation to 2000 steps...

Running batch simulation: 2000 steps
============================================================

Batch 1/200: Native=10, THRML=10 (0.54s, 5.0%)
Batch 2/200: Native=20, THRML=20 (0.52s, 10.0%)
...

✅ DUAL SIMULATION COMPLETE!
   Total time: 1020.3s (17.0 min)
   Ready for parallel history comparison!
   URL: http://localhost:5002/parallel-history.html
```

Then open browser to http://localhost:5002/parallel-history.html and enjoy the synchronized time-travel!
