# Gray-Scott Reaction-Diffusion: Interactive Web Implementations

Two implementations of the Gray-Scott reaction-diffusion model with **time-travel capabilities**:

1. **Native JavaScript** - Client-side simulation with ring buffer history
2. **THRML Server** - Server-side THRML implementation with unlimited history

## Features

- **Time-Travel UI**: Scrub through simulation history
- **Playback Controls**: Play forward/backward, step through frames
- **Interactive Painting**: Click to add chemicals
- **Pattern Presets**: Spots, Stripes, Spirals, Worms
- **Parameter Control**: Adjust F, k, Du, Dv in real-time

---

## 1. Native JavaScript Version

### Quick Start

```bash
cd native-js
# Open index.html in your browser (or use a local server)
python3 -m http.server 8000
# Visit http://localhost:8000
```

### Features
- ✅ Zero latency (runs in browser)
- ✅ Works offline
- ✅ 256x256 grid at 60 FPS
- ✅ Ring buffer stores last 1000 frames
- ✅ No dependencies

### Time-Travel
- History: Last 1000 frames in memory
- Scrubbing: Instant, zero latency
- Limitations: Can't go back beyond ring buffer

---

## 2. THRML Server Version

### Installation

```bash
cd thrml-server
pip install -r requirements.txt
```

### Run Server

```bash
python server.py
```

Server runs on `http://localhost:5000`

### Features
- ✅ Unlimited history (THRML stores all frames)
- ✅ JAX-accelerated computation
- ✅ GPU support (if JAX configured with CUDA)
- ✅ 256x256 grid at ~30 FPS
- ✅ Complete timeline available

### API Endpoints

```
GET  /api/state?step=N      # Get specific timestep
POST /api/params            # Update parameters
POST /api/reset             # Reset simulation
POST /api/interact          # Add chemicals
GET  /api/history           # Get frame range
POST /api/simulate          # Run N steps
```

### Time-Travel
- History: Unlimited (all frames stored)
- Scrubbing: Network latency (~50-100ms)
- Advantages: Can scrub to ANY point since start

---

## Comparison

| Feature | Native JS | THRML Server |
|---------|-----------|--------------|
| **Latency** | 0ms | ~50-100ms |
| **Grid Size** | 256x256 | 256x512+ |
| **History** | 1000 frames | Unlimited |
| **GPU** | No | Yes (JAX) |
| **Deployment** | Static files | Server required |
| **Offline** | Yes | No |
| **Multi-user** | No | Possible |

---

## How It Works

### Gray-Scott Model

The Gray-Scott reaction-diffusion model simulates two chemical species (U and V) interacting:

```
dU/dt = Du * ∇²U - U*V² + F*(1-U)
dV/dt = Dv * ∇²V + U*V² - (F+k)*V
```

Parameters:
- **Du, Dv**: Diffusion rates
- **F**: Feed rate (adds U, removes V)
- **k**: Kill rate (removes V)

Different F and k values produce different patterns!

### Pattern Presets

- **Spots**: F=0.055, k=0.062
- **Stripes**: F=0.035, k=0.060
- **Spirals**: F=0.014, k=0.054
- **Worms**: F=0.039, k=0.058

### THRML Implementation

The THRML version uses:
- **ChemicalU, ChemicalV**: Node types for each species
- **DiffusionFactor**: Spreads chemicals to neighbors
- **ChemicalCouplingFactor**: Links U and V at same location
- **SelfCouplingFactor**: Provides nodes with their own state
- **GrayScottSampler**: Implements reaction-diffusion dynamics

Key advantage: `sample_states()` returns complete history automatically!

---

## Time-Travel Controls

Both implementations feature:

```
[◄] [◄◄] [▶] [▶▶] [►]
├───────●─────────────┤  Timeline scrubber
0     1234         5000
```

- **Play/Pause**: Auto-advance simulation
- **Step**: Single frame forward/backward
- **Scrub**: Drag timeline to any point
- **Speed**: 0.25x, 1x, 2x, 4x playback

---

## Development

### File Structure

```
reaction-diffusion/
├── README.md                  # This file
├── DESIGN.md                  # Architecture docs
├── native-js/
│   └── index.html             # Self-contained JS app
├── thrml-server/
│   ├── server.py              # Flask HTTP server
│   ├── simulation.py          # THRML implementation
│   ├── requirements.txt       # Python deps
│   └── static/
│       └── index.html         # Client UI
└── comparison/
    └── index.html             # Side-by-side (TODO)
```

### Next Steps

- [ ] Comparison UI (side-by-side view)
- [ ] Export to image/video
- [ ] 3D visualization mode
- [ ] Multi-user collaboration
- [ ] WebGL acceleration for JS version
- [ ] Bookmark interesting patterns

---

## Credits

Based on the THRML example: `thrml/examples/04_reaction_diffusion.ipynb`

Gray-Scott model: https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/

---

## License

MIT
