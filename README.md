# Primordial Soup: Evolving Tiny Transformers

A beautiful artificial life simulation where **transformer neural networks evolve from scratch** in a primordial soup. No training, no gradients, no loss functions—just natural selection.

https://github.com/user-attachments/assets/PASTE_VIDEO_ID_HERE

<details>
<summary>📹 How to add the demo video</summary>

1. Push this repo to GitHub
2. Edit this README on GitHub's web interface
3. Drag `assets/demo.mp4` into the text editor
4. GitHub will upload it and generate a URL like `https://github.com/user-attachments/assets/abc123...`
5. Replace the URL above with the generated one

</details>

## The Idea

What if neural networks could evolve like biological life?

In this simulation, **gene fragments** float around in a 2D "primordial soup." When lightning strikes, nearby genes fuse together to form a **genome**—and if that genome is viable, a creature is born.

Each creature has:
- A **genome** (list of genes defining its capabilities)
- A **tiny transformer brain** (architecture determined by genes)
- **Energy** (must eat food or die)
- The ability to **reproduce** (passing genes + weights to offspring)

Creatures that move in interesting patterns survive longer. Creatures that find food can reproduce. Over time, natural selection shapes both the **neural architectures** and the **learned weights**.

## How It Works

### Genes → Brain Architecture

Genes come in four categories:

| Category | Genes | What They Do |
|----------|-------|--------------|
| **INPUT** | `SENSE_FOOD_NEAR`, `SENSE_FOOD_FAR`, `SENSE_ENERGY_SELF`, `SENSE_CREATURES_NEAR`, `SENSE_SIGNAL` | Define what the creature can perceive |
| **OUTPUT** | `MOVE`, `MOVE_FAST`, `EAT`, `EMIT_SIGNAL` | Define what actions the creature can take |
| **ARCHITECTURE** | `EMBED_16/32/64`, `EXTRA_LAYER`, `EXTRA_HEAD`, `LONG_MEMORY` | Shape the transformer's structure |
| **BODY** | `EFFICIENT`, `LARGE_STORAGE`, `FAST_REPRODUCTION` | Physical traits affecting survival |

When genes fuse, they determine the transformer architecture:

```
Genes: [SENSE_FOOD_NEAR, SENSE_FOOD_FAR, MOVE, EAT, EMBED_32, EXTRA_HEAD]
                ↓
        Input dim: 12 (4 + 8 from sensors)
        Output dim: 3 (2 + 1 from actions)
        Embed dim: 32
        Attention heads: 2
        Layers: 1
        Context length: 4
                ↓
        Transformer with ~3,000 parameters
```

### The Transformer Brain

Each creature's brain is a **causal transformer** that maintains a sliding window of past observations:

```
Context Window (4 timesteps):
┌────────────────────────────────────────────┐
│  T-3      T-2      T-1      T(now)        │
│ [obs]    [obs]    [obs]    [obs]          │
└────────────────────────────────────────────┘
                    ↓
            Input Projection
                    ↓
         + Positional Encoding
                    ↓
┌────────────────────────────────────────────┐
│           Self-Attention                   │
│  "Where did I see food before?"           │
│  "Should I keep moving this direction?"   │
└────────────────────────────────────────────┘
                    ↓
          Feed-Forward Network
                    ↓
           Output Projection
                    ↓
        [move_x, move_y, eat, ...]
```

The self-attention mechanism lets creatures "remember" past observations and attend to relevant history when making decisions.

### Dynamical Stability

Not all random neural networks are stable. Some explode to infinity, others collapse to zero. When lightning fuses genes into a creature, we test if the resulting brain is **dynamically stable**:

```python
for _ in range(50):
    output, context = brain(random_input, context)
    if output > 1000 or isnan(output):
        return UNSTABLE  # Creature disintegrates
return STABLE  # Creature lives
```

This is principled: unstable dynamical systems can't sustain coherent behavior. Only stable configurations survive to be tested by natural selection.

### Selection Pressures

Creatures face two selection pressures:

**1. Movement Interest (survival)**
- Interesting patterns (varied directions, turning, circles) → gain energy
- Boring patterns (stationary, straight lines) → lose energy
- This directly selects for visually interesting behavior

**2. Food Finding (reproduction)**
- Creatures start with 70 energy, need 110 to reproduce
- Food gives +40 energy
- Must find food to accumulate enough energy to reproduce

### Reproduction & Inheritance

When a creature has enough energy, it reproduces:

```
Parent (energy ≥ 110) → Reproduce!
        ↓
    Mutate Genome (30% chance):
      - Swap: Replace one gene with random gene
      - Duplicate: Copy one gene
      - Delete: Remove one gene
        ↓
    If architecture unchanged:
      Copy weights + Gaussian noise (σ=0.05)
    Else:
      Initialize new random weights
        ↓
    Child spawns nearby with 55 energy
```

Successful behaviors are inherited: weights that lead to food-finding get passed to offspring with small variations.

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/yourusername/tiny-minds.git
cd tiny-minds

# Create environment and install dependencies
uv sync

# Run the simulation
uv run python src/run.py
```

### Using pip

```bash
git clone https://github.com/yourusername/tiny-minds.git
cd tiny-minds
pip install -r requirements.txt
python src/run.py
```

## Usage

```bash
# Basic run
uv run python src/run.py

# Record frames for GIF export
uv run python src/run.py --record

# Run for specific number of steps
uv run python src/run.py --steps 5000
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `S` | Save screenshot |
| `G` | Export GIF |
| `F` | Toggle gene fragments visibility |
| `ESC` | Quit |
| `Click` | Select creature (view details in sidebar) |

## Configuration

All parameters are in `src/config.py`:

```python
# World
GRID_SIZE = 80              # 80x80 world

# Creatures
MAX_CREATURES = 20          # Population cap
INITIAL_ENERGY = 70         # Starting energy
REPRODUCTION_THRESHOLD = 110 # Energy needed to reproduce

# Food
FOOD_ENERGY = 40            # Energy from eating
MAX_FOOD = 100              # Food cap

# Evolution
MUTATION_CHANCE = 0.30      # Probability of genome mutation
WEIGHT_NOISE_STD = 0.05     # Gaussian noise on inherited weights
```

## The Creatures

Creatures are rendered as cute blob creatures with features determined by their genes:

- **Number of eyes**: Based on sensor genes (1-3 eyes)
- **Eye style**: Normal, sleepy (—), or wide
- **Body bumps**: Architecture genes add bumps
- **Tail**: `LONG_MEMORY` gene adds a tail appendage
- **Color**: Determined by genome hash (similar genomes = similar colors)

## What Emerges?

Over many generations:

1. **Stable architectures** survive (unstable ones die at birth)
2. **Interesting movers** live longer (boring ones starve)
3. **Food finders** reproduce more (random walks into food → more offspring)
4. **Successful weights** spread through the population

The simulation doesn't guarantee intelligence will emerge—but it creates the conditions where it *could*. The transformer architecture provides the capacity for memory and attention; natural selection provides the optimization pressure.

## Project Structure

```
tiny-minds/
├── src/
│   ├── run.py          # Entry point
│   ├── world.py        # Simulation environment
│   ├── creature.py     # Creature class & reproduction
│   ├── brain.py        # Transformer neural network
│   ├── genes.py        # Gene definitions
│   ├── visualize.py    # Pygame rendering
│   └── config.py       # All parameters
├── assets/
│   └── demo.mp4        # Demo video
├── pyproject.toml      # Project config (uv)
├── requirements.txt    # Dependencies (pip fallback)
└── README.md
```

## Philosophy

This project explores a fundamental question: **Can neural network architectures and weights co-evolve through natural selection alone?**

| Traditional ML | This Project |
|----------------|--------------|
| Design architecture | Random architectures |
| Collect data | No data |
| Train with gradients | No gradients |
| Optimize loss function | Natural selection |

Key insights:
- **Stability as a filter**: Dynamical stability is a necessary (not sufficient) condition for intelligent behavior
- **Architecture-weight coupling**: The same weights mean different things in different architectures
- **Selection pressure design**: What you select for is what you get

## Future Directions

- [ ] Predator-prey dynamics
- [ ] Sexual reproduction (two parents)
- [ ] More complex environments
- [ ] Longer evolution runs with checkpointing
- [ ] Lineage tracking and visualization

## License

MIT

---

*"In the beginning, there was only soup. Then lightning struck."*
