"""
Configuration for the Primordial Soup simulation.
All the numbers that control how the simulation behaves live here.
"""

# =============================================================================
# WORLD SETTINGS
# =============================================================================

GRID_SIZE = 80                   # World is 80x80 tiles (fits nicely with larger cells)
WRAP_AROUND = True               # Edges connect (like a donut/torus)

# =============================================================================
# GENE FRAGMENT SETTINGS
# =============================================================================

INITIAL_FRAGMENTS = 80           # How many gene fragments start in the soup
FRAGMENT_SPAWN_RATE = 1          # New fragments appear per timestep
FRAGMENT_DRIFT_SPEED = 0.2       # How fast fragments float around (Brownian motion)

# =============================================================================
# LIGHTNING SETTINGS
# =============================================================================

LIGHTNING_CHANCE = 0.15          # Probability of lightning each timestep
GENES_PER_STRIKE = (6, 12)       # Lightning grabs 6-12 nearby genes
LIGHTNING_GATHER_RADIUS = 15.0   # How far lightning reaches for genes

# =============================================================================
# FOOD SETTINGS
# =============================================================================

INITIAL_FOOD = 50                # Food tiles at start
FOOD_SPAWN_RATE = 2              # New food tiles per timestep
MAX_FOOD = 100                   # Cap on total food in world
FOOD_ENERGY = 40                 # Energy gained from eating food

# =============================================================================
# CREATURE SETTINGS
# =============================================================================

INITIAL_ENERGY = 70              # Energy when creature is born from lightning
MAX_ENERGY_BASE = 150            # Base maximum energy storage
BASE_ENERGY_COST = 0.25          # Energy cost just for existing each step
ACTION_ENERGY_COST = 0.1         # Extra cost per action taken (low - we want movement!)
MOVE_FAST_MULTIPLIER = 2.0       # Fast movement costs this much more
MOVE_THRESHOLD = 0.02            # How much output needed to trigger movement (very low!)

REPRODUCTION_THRESHOLD = 110     # Need this much energy to reproduce (need food!)
REPRODUCTION_COST = 50           # Energy spent making a child
CHILD_ENERGY = 55                # Energy the child starts with

RANDOM_DEATH_CHANCE = 0.001      # Small chance of random death (prevents immortals)

# =============================================================================
# MUTATION SETTINGS
# =============================================================================

MUTATION_CHANCE = 0.30           # Chance of any mutation when reproducing
MUTATION_WEIGHTS = {             # Relative probabilities of mutation types
    'none': 0.70,                # No mutation
    'swap': 0.15,                # Replace one gene with another
    'duplicate': 0.08,           # Copy one gene
    'delete': 0.07,              # Remove one gene
}
WEIGHT_NOISE_STD = 0.05          # Noise added to child's neural network weights

# =============================================================================
# NEURAL NETWORK / STABILITY SETTINGS
# =============================================================================

STABILITY_TEST_STEPS = 50        # How many forward passes to test stability
STABILITY_THRESHOLD = 1000.0     # Activations above this = unstable
DEFAULT_HIDDEN_DIM = 16          # If no hidden genes, use this

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

WINDOW_WIDTH = 900               # Window width in pixels
WINDOW_HEIGHT = 640              # Window height in pixels (matches grid: 80 * 8)
CELL_SIZE = 8                    # Pixels per grid cell (larger = bigger creatures)
FPS = 24                         # Slower for more relaxed feel
SHOW_FRAGMENTS = False           # Gene fragments hidden by default (press F to show)
FRAGMENT_ALPHA = 60              # Transparency of fragments (0-255, lower = more transparent)

# Colors (R, G, B)
COLOR_BACKGROUND = (15, 15, 25)           # Dark blue-black
COLOR_FOOD = (50, 205, 50)                # Lime green
COLOR_LIGHTNING = (255, 255, 200)         # Bright yellow-white
COLOR_TEXT = (220, 220, 220)              # Light gray
COLOR_PANEL = (25, 25, 40)                # Darker panel background

# Gene fragment colors by category
GENE_COLORS = {
    'input': (100, 149, 237),             # Cornflower blue
    'output': (255, 127, 80),             # Coral
    'architecture': (186, 85, 211),       # Medium orchid
    'body': (60, 179, 113),               # Medium sea green
}

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

MAX_CREATURES = 20               # Small population - each creature clearly visible
LOG_INTERVAL = 100               # Print stats every N steps
