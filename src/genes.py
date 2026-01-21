"""
Genes - The building blocks of creature DNA.

Each gene is a simple instruction that affects what a creature can do.
Genes come in four types:
  - INPUT genes:  What the creature can sense (eyes, ears, etc.)
  - OUTPUT genes: What the creature can do (move, eat, signal)
  - ARCHITECTURE genes: How big/complex the creature's brain is
  - BODY genes: Physical traits (efficiency, storage, etc.)
"""

from enum import Enum, auto
from dataclasses import dataclass
import random


class GeneCategory(Enum):
    """The four categories of genes."""
    INPUT = auto()        # Sensing
    OUTPUT = auto()       # Acting
    ARCHITECTURE = auto() # Brain structure
    BODY = auto()         # Physical traits


@dataclass(frozen=True)
class GeneType:
    """
    Definition of a type of gene.

    Attributes:
        name: Human-readable name
        category: Which category this gene belongs to
        dimensions: For INPUT/OUTPUT genes, how many numbers it adds
        description: What this gene does
    """
    name: str
    category: GeneCategory
    dimensions: int = 0
    description: str = ""

    def __hash__(self):
        return hash(self.name)


# =============================================================================
# DEFINE ALL GENE TYPES
# =============================================================================

# Input genes - what creatures can sense
SENSE_FOOD_NEAR = GeneType(
    name="SENSE_FOOD_NEAR",
    category=GeneCategory.INPUT,
    dimensions=4,
    description="See food in 4 adjacent tiles (up/down/left/right)"
)

SENSE_FOOD_FAR = GeneType(
    name="SENSE_FOOD_FAR",
    category=GeneCategory.INPUT,
    dimensions=8,
    description="See food in 8 surrounding tiles (larger radius)"
)

SENSE_ENERGY_SELF = GeneType(
    name="SENSE_ENERGY_SELF",
    category=GeneCategory.INPUT,
    dimensions=1,
    description="Know your own energy level"
)

SENSE_CREATURES_NEAR = GeneType(
    name="SENSE_CREATURES_NEAR",
    category=GeneCategory.INPUT,
    dimensions=4,
    description="Detect other creatures in 4 adjacent tiles"
)

SENSE_SIGNAL = GeneType(
    name="SENSE_SIGNAL",
    category=GeneCategory.INPUT,
    dimensions=4,
    description="Detect signals from nearby creatures"
)

# Output genes - what creatures can do
MOVE = GeneType(
    name="MOVE",
    category=GeneCategory.OUTPUT,
    dimensions=2,
    description="Move in x/y direction"
)

MOVE_FAST = GeneType(
    name="MOVE_FAST",
    category=GeneCategory.OUTPUT,
    dimensions=1,
    description="Toggle to move faster (costs more energy)"
)

EAT = GeneType(
    name="EAT",
    category=GeneCategory.OUTPUT,
    dimensions=1,
    description="Attempt to eat food on current tile"
)

EMIT_SIGNAL = GeneType(
    name="EMIT_SIGNAL",
    category=GeneCategory.OUTPUT,
    dimensions=1,
    description="Broadcast a signal to nearby creatures"
)

# Architecture genes - transformer brain structure
EMBED_16 = GeneType(
    name="EMBED_16",
    category=GeneCategory.ARCHITECTURE,
    description="Transformer embedding dimension 16"
)

EMBED_32 = GeneType(
    name="EMBED_32",
    category=GeneCategory.ARCHITECTURE,
    description="Transformer embedding dimension 32"
)

EMBED_64 = GeneType(
    name="EMBED_64",
    category=GeneCategory.ARCHITECTURE,
    description="Transformer embedding dimension 64"
)

EXTRA_LAYER = GeneType(
    name="EXTRA_LAYER",
    category=GeneCategory.ARCHITECTURE,
    description="Add an extra transformer layer"
)

EXTRA_HEAD = GeneType(
    name="EXTRA_HEAD",
    category=GeneCategory.ARCHITECTURE,
    description="Add an extra attention head"
)

LONG_MEMORY = GeneType(
    name="LONG_MEMORY",
    category=GeneCategory.ARCHITECTURE,
    description="Longer context window (more memory)"
)

# Body genes - physical traits
EFFICIENT = GeneType(
    name="EFFICIENT",
    category=GeneCategory.BODY,
    description="Use 10% less energy"
)

LARGE_STORAGE = GeneType(
    name="LARGE_STORAGE",
    category=GeneCategory.BODY,
    description="Can store 50% more energy"
)

FAST_REPRODUCTION = GeneType(
    name="FAST_REPRODUCTION",
    category=GeneCategory.BODY,
    description="Need 20% less energy to reproduce"
)


# =============================================================================
# GENE LISTS AND HELPERS
# =============================================================================

# All input genes
INPUT_GENES = [
    SENSE_FOOD_NEAR,
    SENSE_FOOD_FAR,
    SENSE_ENERGY_SELF,
    SENSE_CREATURES_NEAR,
    SENSE_SIGNAL,
]

# All output genes
OUTPUT_GENES = [
    MOVE,
    MOVE_FAST,
    EAT,
    EMIT_SIGNAL,
]

# All architecture genes
ARCHITECTURE_GENES = [
    EMBED_16,
    EMBED_32,
    EMBED_64,
    EXTRA_LAYER,
    EXTRA_HEAD,
    LONG_MEMORY,
]

# All body genes
BODY_GENES = [
    EFFICIENT,
    LARGE_STORAGE,
    FAST_REPRODUCTION,
]

# Every gene type in one list
ALL_GENES = INPUT_GENES + OUTPUT_GENES + ARCHITECTURE_GENES + BODY_GENES

# Weights for random gene selection (some genes are rarer)
# Higher weight = more common in the soup
GENE_WEIGHTS = {
    # Input genes - SENSE_FOOD is most important for survival
    SENSE_FOOD_NEAR: 15,
    SENSE_FOOD_FAR: 8,
    SENSE_ENERGY_SELF: 6,
    SENSE_CREATURES_NEAR: 4,
    SENSE_SIGNAL: 2,
    # Output genes - MOVE and EAT are essential for survival!
    MOVE: 20,           # Very common - creatures need to move!
    MOVE_FAST: 3,
    EAT: 18,            # Very common - creatures need to eat!
    EMIT_SIGNAL: 2,
    # Architecture genes - transformer structure
    EMBED_16: 6,        # Small embedding
    EMBED_32: 4,        # Medium embedding
    EMBED_64: 2,        # Large embedding
    EXTRA_LAYER: 2,     # More transformer layers
    EXTRA_HEAD: 3,      # More attention heads
    LONG_MEMORY: 3,     # Longer context window
    # Body genes - moderately common
    EFFICIENT: 5,
    LARGE_STORAGE: 3,
    FAST_REPRODUCTION: 3,
}


def random_gene() -> GeneType:
    """Pick a random gene, weighted by rarity."""
    genes = list(GENE_WEIGHTS.keys())
    weights = list(GENE_WEIGHTS.values())
    return random.choices(genes, weights=weights)[0]


def get_gene_color(gene: GeneType) -> tuple:
    """Get the display color for a gene based on its category."""
    from config import GENE_COLORS
    return GENE_COLORS[gene.category.name.lower()]


@dataclass
class Gene:
    """
    A single gene instance.
    This is what actually exists in genomes and floats in the soup.
    """
    gene_type: GeneType

    @property
    def name(self) -> str:
        return self.gene_type.name

    @property
    def category(self) -> GeneCategory:
        return self.gene_type.category

    def __repr__(self):
        return f"Gene({self.gene_type.name})"
