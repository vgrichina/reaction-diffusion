# Quick Start Guide

## Option 1: Native JavaScript (Easiest!)

No installation required - just open in your browser:

```bash
cd native-js
open index.html
# Or use a local server:
python3 -m http.server 8000
# Then visit: http://localhost:8000
```

**Features:**
- Zero latency
- Works offline
- Time-travel with 1000 frame buffer
- Drag timeline to scrub through history

---

## Option 2: THRML Server (Full Power!)

### Install Dependencies

```bash
cd thrml-server
pip install -r requirements.txt
```

### Run Server

```bash
python server.py
```

Server starts on `http://localhost:5000`

**Features:**
- Unlimited time-travel history
- JAX-accelerated (GPU support)
- Complete timeline always available

---

## Option 3: Side-by-Side Comparison

### Terminal 1: Start THRML Server

```bash
cd thrml-server
python server.py
```

### Terminal 2: Serve Comparison Page

```bash
cd comparison
python3 -m http.server 8001
```

Visit: `http://localhost:8001`

**See both implementations running in parallel!**

---

## Time-Travel Controls

All implementations feature:

```
[◄] Step Back
[◄◄] Play Backward
[▶] Play/Pause
[▶▶] Play Forward
[►] Step Forward

Drag timeline slider to scrub to any point
```

---

## Pattern Presets

Try these parameter combinations:

- **Spots**: F=0.055, k=0.062 (Classic Turing patterns)
- **Stripes**: F=0.035, k=0.060 (Parallel lines)
- **Spirals**: F=0.014, k=0.054 (Rotating patterns)
- **Worms**: F=0.039, k=0.058 (Squirming structures)

---

## Tips

1. **Native JS**: Best for immediate experimentation
2. **THRML Server**: Best for exploring full history
3. **Comparison**: Best for understanding differences

## Troubleshooting

**THRML server not working?**
- Make sure you're in `thrml-server` directory
- Check THRML is installed: `cd ../../thrml && pip install -e .`
- Check dependencies: `pip install -r requirements.txt`

**Slow performance?**
- Native JS: Try smaller brush size
- THRML: Check JAX installation for GPU support
