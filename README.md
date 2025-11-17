# Gray-Scott Reaction-Diffusion with THRML

Interactive web-based Gray-Scott reaction-diffusion simulation powered by **THRML** (Thermal), showcasing unlimited history and perfect time-travel capabilities.

## Features

- **Unlimited History**: Every simulation step is permanently stored
- **Perfect Time-Travel**: Scrub through the entire simulation timeline
- **JAX-Powered**: Hardware-accelerated computation with JIT compilation
- **Factor Graph Model**: Probabilistic graphical model implementation
- **Interactive Controls**: Real-time parameter adjustment and pattern exploration
- **Playback Controls**: Play, pause, step through frames at any speed

---

## Quick Start

### Installation

```bash
cd thrml-server
pip install -r requirements.txt
```

### Run Server

```bash
python server.py
```

Server runs on `http://localhost:5001`

Open your browser and start exploring!

---

## What is THRML?

**THRML (Thermal)** is a probabilistic programming framework built on JAX that uses factor graphs and Gibbs sampling to model complex systems. This demo showcases how THRML naturally maintains complete simulation history as a byproduct of its sampling process.

### Key Advantages

- ✅ **Automatic History Tracking**: `sample_states()` returns complete timeline
- ✅ **JAX Acceleration**: GPU support for faster computation
- ✅ **Probabilistic Framework**: Factor graph representation
- ✅ **Scalable**: Handles large grids efficiently

---

## The Gray-Scott Model

The Gray-Scott reaction-diffusion model simulates two chemical species (U and V) interacting:

```
dU/dt = Du * ∇²U - U*V² + F*(1-U)
dV/dt = Dv * ∇²V + U*V² - (F+k)*V
```

**Parameters:**
- **Du, Dv**: Diffusion rates (how fast chemicals spread)
- **F**: Feed rate (adds U, removes V)
- **k**: Kill rate (removes V)

Different F and k values produce fascinating patterns!

### Pattern Presets

Try these parameter combinations:

- **Spots**: F=0.055, k=0.062 (Classic Turing patterns)
- **Stripes**: F=0.035, k=0.060 (Parallel lines)
- **Spirals**: F=0.014, k=0.054 (Rotating patterns)
- **Worms**: F=0.039, k=0.058 (Squirming structures)

---

## API Endpoints

The THRML server provides a RESTful API:

```
GET  /                      # Playground interface
GET  /api/state?step=N      # Get specific timestep
POST /api/params            # Update parameters (F, k, Du, Dv)
POST /api/reset             # Reset simulation with pattern
POST /api/interact          # Add chemicals at position
POST /api/simulate          # Run N steps
POST /api/pause             # Pause background simulation
POST /api/resume            # Resume background simulation
```

---

## THRML Implementation

The simulation uses THRML's factor graph framework:

- **ChemicalU, ChemicalV**: Node types for each chemical species
- **DiffusionFactor**: Spreads chemicals to neighboring cells
- **ChemicalCouplingFactor**: Links U and V at the same location
- **SelfCouplingFactor**: Provides nodes with their own previous state
- **GrayScottSampler**: Implements the reaction-diffusion dynamics

**Key Innovation**: The `sample_states()` function naturally returns the complete simulation history, making time-travel a built-in feature rather than an add-on.

---

## Time-Travel Controls

The playground features intuitive timeline controls:

```
[◄] [▶] [►]
├───────●─────────────┤  Timeline scrubber
0     step         max
```

- **Play/Pause**: Auto-advance simulation
- **Step Back/Forward**: Single frame navigation
- **Scrubber**: Drag to any point in history
- **Jump to Start/End**: Instant navigation

---

## File Structure

```
reaction-diffusion/
├── README.md                  # This file
├── QUICKSTART.md              # Quick start guide
├── IMPLEMENTATION_SUMMARY.md  # Technical details
├── demo.sh                    # Launch script
└── thrml-server/
    ├── server.py              # Flask HTTP server
    ├── simulation.py          # THRML implementation
    ├── requirements.txt       # Python dependencies
    └── static/
        └── thrml-playground.html  # Interactive UI
```

---

## Development

### Running with Custom Port

```bash
PORT=8080 python server.py
```

### Grid Size

The default grid is 128x128. To change it, modify `server.py`:

```python
session = SimulationSession(256, 256)  # Larger grid
```

---

## Credits

Based on the THRML framework and the classic Gray-Scott reaction-diffusion model.

- THRML: Probabilistic programming with factor graphs
- Gray-Scott Model: https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/

---

## License

MIT
