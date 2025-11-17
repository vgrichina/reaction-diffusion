# Quick Start Guide

Get up and running with the THRML Gray-Scott playground in minutes!

## Installation

```bash
cd thrml-server
pip install -r requirements.txt
```

## Run the Server

```bash
python server.py
```

The server will start on `http://localhost:5001`

## Using the Playground

### 1. Choose a Pattern

Click one of the initial pattern buttons:
- 📏 Stripes
- ⬛ Center Square
- 🎲 Random
- ⭕ Circle

### 2. Select Parameters

Try one of the preset parameter combinations:
- **Spots**: Creates classic Turing spot patterns
- **Stripes**: Generates parallel stripe patterns
- **Spirals**: Produces rotating spiral patterns
- **Worms**: Creates worm-like moving structures

Or use the sliders to experiment with custom F (feed rate) and k (kill rate) values.

### 3. Reset & Run

1. Click **"Reset Simulation"** to apply your chosen pattern and parameters
2. Click **"Run 100 Steps"** to compute 100 simulation steps
3. Watch the pattern evolve!

### 4. Time Travel

Use the timeline controls to explore the simulation history:

```
[◄] Step Back
[▶] Play/Pause (auto-advance)
[►] Step Forward
[⏮ Start] Jump to beginning
[⏭ End] Jump to latest
```

Drag the timeline slider to scrub to any point in the history!

---

## Tips

- **Experiment freely**: Try different F and k combinations
- **Use time-travel**: Scrub back to see how patterns evolved
- **Run multiple batches**: Click "Run 100 Steps" multiple times to build longer histories
- **Watch in real-time**: Use the Play button to auto-advance through computed steps

---

## Pattern Presets Explained

| Pattern | F | k | What You'll See |
|---------|---|---|-----------------|
| **Spots** | 0.055 | 0.062 | Circular spots that form and stabilize |
| **Stripes** | 0.035 | 0.060 | Parallel lines across the grid |
| **Spirals** | 0.014 | 0.054 | Rotating spiral patterns |
| **Worms** | 0.039 | 0.058 | Squirming, elongated structures |

---

## Troubleshooting

**Server won't start?**
- Make sure you're in the `thrml-server` directory
- Check THRML is installed: `pip install thrml`
- Verify dependencies: `pip install -r requirements.txt`

**Slow performance?**
- The server is computing 100 steps when you click the button
- JAX will JIT compile on first run (initial delay is normal)
- For GPU acceleration, install JAX with CUDA support

**Port already in use?**
```bash
PORT=8080 python server.py
```

---

## What Makes This Special?

Unlike typical simulations, THRML's `sample_states()` function **automatically preserves the entire simulation history**. This means:

- ✅ Perfect time-travel to any previous step
- ✅ No ring buffer limitations
- ✅ Complete timeline always available
- ✅ Scrub through thousands of steps instantly

This is the power of THRML's probabilistic framework!
