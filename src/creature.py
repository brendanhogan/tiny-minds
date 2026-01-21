"""
Creature - Living beings in the primordial soup.

Each creature has:
  - A genome (list of genes that define it)
  - A brain (tiny transformer for decision making)
  - Position in the world
  - Energy (fuel for staying alive)
  - Context memory (past observations for the transformer)
"""

import random
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import config
from genes import (
    Gene, GeneType, GeneCategory, random_gene,
    SENSE_FOOD_NEAR, SENSE_FOOD_FAR, SENSE_ENERGY_SELF,
    SENSE_CREATURES_NEAR, SENSE_SIGNAL,
    MOVE, MOVE_FAST, EAT, EMIT_SIGNAL,
    EMBED_16, EMBED_32, EMBED_64, EXTRA_LAYER, EXTRA_HEAD, LONG_MEMORY,
    EFFICIENT, LARGE_STORAGE, FAST_REPRODUCTION,
    INPUT_GENES, OUTPUT_GENES
)
from brain import Brain, test_brain_stability


@dataclass
class Genome:
    """
    A creature's complete genetic code.
    Just a list of genes that together define the creature.
    """
    genes: List[Gene] = field(default_factory=list)

    def get_genes_of_type(self, gene_type: GeneType) -> List[Gene]:
        """Get all genes of a specific type."""
        return [g for g in self.genes if g.gene_type == gene_type]

    def get_genes_in_category(self, category: GeneCategory) -> List[Gene]:
        """Get all genes in a category (INPUT, OUTPUT, etc.)."""
        return [g for g in self.genes if g.category == category]

    def has_gene(self, gene_type: GeneType) -> bool:
        """Check if this genome has at least one gene of this type."""
        return any(g.gene_type == gene_type for g in self.genes)

    def count_gene(self, gene_type: GeneType) -> int:
        """Count how many of this gene type we have."""
        return sum(1 for g in self.genes if g.gene_type == gene_type)

    def get_input_dim(self) -> int:
        """Calculate total input dimensions from input genes."""
        total = 0
        for g in self.get_genes_in_category(GeneCategory.INPUT):
            total += g.gene_type.dimensions
        return total

    def get_output_dim(self) -> int:
        """Calculate total output dimensions from output genes."""
        total = 0
        for g in self.get_genes_in_category(GeneCategory.OUTPUT):
            total += g.gene_type.dimensions
        return total

    # =========================================================================
    # Transformer architecture from genes
    # =========================================================================

    def get_embed_dim(self) -> int:
        """Determine transformer embedding dimension from architecture genes."""
        # Largest embedding gene wins, must be divisible by num_heads
        if self.has_gene(EMBED_64):
            return 64
        elif self.has_gene(EMBED_32):
            return 32
        elif self.has_gene(EMBED_16):
            return 16
        else:
            return 16  # Default small embedding

    def get_num_layers(self) -> int:
        """Count transformer layers (1 base + extra layer genes, max 3)."""
        extra = self.count_gene(EXTRA_LAYER)
        return min(1 + extra, 3)

    def get_num_heads(self) -> int:
        """Determine number of attention heads."""
        # Base is 1, extra head genes add more (max limited by embed_dim)
        base = 1
        extra = self.count_gene(EXTRA_HEAD)
        num_heads = base + extra

        # Must divide embed_dim evenly
        embed_dim = self.get_embed_dim()
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        return num_heads

    def get_context_len(self) -> int:
        """Determine context window length (memory)."""
        # Base is 4 timesteps, LONG_MEMORY adds more
        base = 4
        extra = self.count_gene(LONG_MEMORY) * 4
        return min(base + extra, 16)  # Cap at 16

    # =========================================================================
    # Body stats from genes
    # =========================================================================

    def get_efficiency_bonus(self) -> float:
        """Calculate energy efficiency from EFFICIENT genes."""
        count = self.count_gene(EFFICIENT)
        # Diminishing returns: 10%, 18%, 24%, 28%...
        bonus = 0.0
        for i in range(count):
            bonus += 0.10 * (0.8 ** i)
        return min(bonus, 0.5)  # Cap at 50% reduction

    def get_storage_bonus(self) -> float:
        """Calculate extra energy storage from LARGE_STORAGE genes."""
        count = self.count_gene(LARGE_STORAGE)
        return count * 0.5  # 50% more per gene

    def get_reproduction_bonus(self) -> float:
        """Calculate reproduction threshold reduction."""
        count = self.count_gene(FAST_REPRODUCTION)
        # Diminishing returns
        bonus = 0.0
        for i in range(count):
            bonus += 0.20 * (0.7 ** i)
        return min(bonus, 0.5)  # Cap at 50% reduction

    # =========================================================================
    # Mutation
    # =========================================================================

    def mutate(self) -> 'Genome':
        """
        Create a mutated copy of this genome.

        Possible mutations:
          - swap: Replace one gene with a different type
          - duplicate: Copy one gene
          - delete: Remove one gene
        """
        new_genes = [Gene(g.gene_type) for g in self.genes]

        if random.random() < config.MUTATION_CHANCE:
            mutation_type = random.choices(
                list(config.MUTATION_WEIGHTS.keys()),
                list(config.MUTATION_WEIGHTS.values())
            )[0]

            if mutation_type == 'swap' and new_genes:
                # Replace a random gene with a new random gene
                idx = random.randrange(len(new_genes))
                new_genes[idx] = Gene(random_gene())

            elif mutation_type == 'duplicate' and new_genes:
                # Duplicate a random gene
                idx = random.randrange(len(new_genes))
                new_genes.append(Gene(new_genes[idx].gene_type))

            elif mutation_type == 'delete' and len(new_genes) > 1:
                # Remove a random gene (keep at least 1)
                idx = random.randrange(len(new_genes))
                new_genes.pop(idx)

        return Genome(genes=new_genes)

    # =========================================================================
    # Utility
    # =========================================================================

    def signature(self) -> str:
        """Create a simple signature string for this genome."""
        # Sort genes by name for consistent ordering
        names = sorted(g.name for g in self.genes)
        return "-".join(names)

    def short_signature(self) -> str:
        """Create a very short signature (for display)."""
        # Count each gene type
        counts = {}
        for g in self.genes:
            name = g.name[:3]  # First 3 chars
            counts[name] = counts.get(name, 0) + 1

        parts = [f"{k}{v}" if v > 1 else k for k, v in sorted(counts.items())]
        return "".join(parts)

    def __hash__(self):
        return hash(self.signature())

    def __repr__(self):
        return f"Genome({len(self.genes)} genes)"


@dataclass
class Creature:
    """
    A living creature in the primordial soup.
    """
    genome: Genome
    brain: Brain
    x: int
    y: int
    energy: float
    age: int = 0
    context: Optional[torch.Tensor] = None  # Memory for transformer
    is_signaling: bool = False  # Currently emitting a signal?

    # For visualization - unique ID and color
    creature_id: int = 0
    _id_counter: int = 0

    def __post_init__(self):
        # Assign unique ID
        Creature._id_counter += 1
        self.creature_id = Creature._id_counter

    @property
    def max_energy(self) -> float:
        """Maximum energy this creature can store."""
        bonus = self.genome.get_storage_bonus()
        return config.MAX_ENERGY_BASE * (1 + bonus)

    @property
    def reproduction_threshold(self) -> float:
        """Energy needed to reproduce."""
        bonus = self.genome.get_reproduction_bonus()
        return config.REPRODUCTION_THRESHOLD * (1 - bonus)

    @property
    def efficiency(self) -> float:
        """Energy cost multiplier (lower = more efficient)."""
        bonus = self.genome.get_efficiency_bonus()
        return 1.0 - bonus

    def get_color(self) -> Tuple[int, int, int]:
        """Generate a color based on genome (similar genomes = similar colors)."""
        # Use hash of genome signature to generate consistent color
        h = hash(self.genome.signature())
        r = ((h >> 16) & 0xFF)
        g = ((h >> 8) & 0xFF)
        b = (h & 0xFF)
        # Make sure it's not too dark
        min_brightness = 100
        r = min_brightness + (r * (255 - min_brightness)) // 255
        g = min_brightness + (g * (255 - min_brightness)) // 255
        b = min_brightness + (b * (255 - min_brightness)) // 255
        return (r, g, b)

    def can_reproduce(self) -> bool:
        """Check if creature has enough energy to reproduce."""
        return self.energy >= self.reproduction_threshold

    def reproduce(self) -> Optional['Creature']:
        """
        Create a child creature.
        Returns None if reproduction fails.
        """
        if not self.can_reproduce():
            return None

        # Pay the cost
        self.energy -= config.REPRODUCTION_COST

        # Mutate genome
        child_genome = self.genome.mutate()

        # Check if child is viable
        input_dim = child_genome.get_input_dim()
        output_dim = child_genome.get_output_dim()

        if input_dim == 0 or output_dim == 0:
            # Child can't sense or act - not viable
            return None

        # Get child's architecture
        child_embed_dim = child_genome.get_embed_dim()
        child_num_heads = child_genome.get_num_heads()
        child_num_layers = child_genome.get_num_layers()
        child_context_len = child_genome.get_context_len()

        # Check if architecture matches parent
        same_architecture = (
            input_dim == self.brain.input_dim and
            output_dim == self.brain.output_dim and
            child_embed_dim == self.brain.embed_dim and
            child_num_heads == self.brain.num_heads and
            child_num_layers == self.brain.num_layers and
            child_context_len == self.brain.context_len
        )

        if same_architecture:
            # Same architecture - inherit weights with noise
            child_brain = self.brain.copy_with_noise(config.WEIGHT_NOISE_STD)
        else:
            # Different architecture - new random brain
            child_brain = Brain(
                input_dim=input_dim,
                output_dim=output_dim,
                embed_dim=child_embed_dim,
                num_heads=child_num_heads,
                num_layers=child_num_layers,
                context_len=child_context_len
            )

        # Test stability
        if not test_brain_stability(child_brain):
            return None

        # Spawn near parent (but not exactly on top)
        # Pick a random direction and spawn 2-3 tiles away
        dx = random.choice([-2, -1, 1, 2])
        dy = random.choice([-2, -1, 1, 2])
        child_x = (self.x + dx) % config.GRID_SIZE
        child_y = (self.y + dy) % config.GRID_SIZE

        return Creature(
            genome=child_genome,
            brain=child_brain,
            x=child_x,
            y=child_y,
            energy=config.CHILD_ENERGY,
            age=0
        )


def create_creature_from_genes(genes: List[Gene], x: int, y: int) -> Optional[Creature]:
    """
    Attempt to create a creature from a list of genes.
    Returns None if the creature is not viable (unstable or can't act).
    """
    genome = Genome(genes=genes)

    # Check for minimum viability
    input_dim = genome.get_input_dim()
    output_dim = genome.get_output_dim()

    if input_dim == 0:
        # Can't sense anything - not viable
        return None

    if output_dim == 0:
        # Can't do anything - not viable
        return None

    # Build the transformer brain
    brain = Brain(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=genome.get_embed_dim(),
        num_heads=genome.get_num_heads(),
        num_layers=genome.get_num_layers(),
        context_len=genome.get_context_len()
    )

    # Test stability
    if not test_brain_stability(brain):
        return None

    # Success! Create the creature
    return Creature(
        genome=genome,
        brain=brain,
        x=x,
        y=y,
        energy=config.INITIAL_ENERGY
    )
