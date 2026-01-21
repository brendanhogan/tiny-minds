"""
World - The primordial soup environment.

The world contains:
  - A 2D grid where creatures live
  - Food that spawns randomly
  - Gene fragments floating around
  - Lightning that fuses genes into creatures
"""

import random
import math
import torch
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass, field

import config
from genes import Gene, GeneCategory, random_gene, INPUT_GENES, OUTPUT_GENES
from genes import SENSE_FOOD_NEAR, SENSE_FOOD_FAR, SENSE_ENERGY_SELF
from genes import SENSE_CREATURES_NEAR, SENSE_SIGNAL
from genes import MOVE, MOVE_FAST, EAT, EMIT_SIGNAL
from creature import Creature, create_creature_from_genes


@dataclass
class GeneFragment:
    """A gene fragment floating in the soup."""
    gene: Gene
    x: float  # Continuous position for smooth movement
    y: float
    vx: float = 0.0  # Velocity for Brownian motion
    vy: float = 0.0

    def update_position(self):
        """Move with Brownian motion."""
        # Random acceleration (Brownian motion)
        self.vx += random.gauss(0, 0.3)
        self.vy += random.gauss(0, 0.3)

        # Damping
        self.vx *= 0.9
        self.vy *= 0.9

        # Speed limit
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > config.FRAGMENT_DRIFT_SPEED:
            self.vx = self.vx / speed * config.FRAGMENT_DRIFT_SPEED
            self.vy = self.vy / speed * config.FRAGMENT_DRIFT_SPEED

        # Move
        self.x += self.vx
        self.y += self.vy

        # Wrap around
        self.x = self.x % config.GRID_SIZE
        self.y = self.y % config.GRID_SIZE


@dataclass
class LightningStrike:
    """Record of a lightning strike for visualization."""
    x: int
    y: int
    age: int = 0  # How many frames ago it happened
    success: bool = False  # Did it create a creature?


class World:
    """The primordial soup - where life emerges."""

    def __init__(self):
        # The grid
        self.size = config.GRID_SIZE

        # Living creatures
        self.creatures: List[Creature] = []

        # Food locations
        self.food: Set[Tuple[int, int]] = set()

        # Track creature position history for stagnation detection
        # creature_id -> list of (x, y) positions
        self.position_history: Dict[int, List[Tuple[int, int]]] = {}

        # Floating gene fragments
        self.fragments: List[GeneFragment] = []

        # Signals being emitted (position -> intensity)
        self.signals: Dict[Tuple[int, int], float] = {}

        # Recent lightning strikes (for visualization)
        self.lightning_strikes: List[LightningStrike] = []

        # Statistics
        self.step_count = 0
        self.total_births = 0
        self.total_deaths = 0
        self.total_lightning = 0
        self.successful_lightning = 0

        # Initialize
        self._spawn_initial_fragments()
        self._spawn_initial_food()

    def _spawn_initial_fragments(self):
        """Scatter initial gene fragments across the world."""
        for _ in range(config.INITIAL_FRAGMENTS):
            gene = Gene(random_gene())
            x = random.random() * self.size
            y = random.random() * self.size
            self.fragments.append(GeneFragment(gene=gene, x=x, y=y))

    def _spawn_initial_food(self):
        """Place initial food randomly."""
        while len(self.food) < config.INITIAL_FOOD:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            self.food.add((x, y))

    def step(self):
        """Advance the simulation by one timestep."""
        self.step_count += 1

        # Update gene fragments (Brownian motion)
        for fragment in self.fragments:
            fragment.update_position()

        # Maybe spawn new fragments
        for _ in range(config.FRAGMENT_SPAWN_RATE):
            gene = Gene(random_gene())
            x = random.random() * self.size
            y = random.random() * self.size
            self.fragments.append(GeneFragment(gene=gene, x=x, y=y))

        # Maybe lightning strikes
        if random.random() < config.LIGHTNING_CHANCE:
            self._lightning_strike()

        # Update all creatures
        self._update_creatures()

        # Spawn food
        self._spawn_food()

        # Clear old signals
        self.signals = {}
        for creature in self.creatures:
            if creature.is_signaling:
                self.signals[(creature.x, creature.y)] = 1.0

        # Age lightning strikes and remove old ones
        self.lightning_strikes = [
            LightningStrike(s.x, s.y, s.age + 1, s.success)
            for s in self.lightning_strikes if s.age < 15
        ]

    def _lightning_strike(self):
        """Lightning hits a random location and fuses nearby genes."""
        self.total_lightning += 1

        # Random strike location
        strike_x = random.randint(0, self.size - 1)
        strike_y = random.randint(0, self.size - 1)

        # How many genes to gather
        num_genes = random.randint(*config.GENES_PER_STRIKE)

        # Find nearest fragments
        def distance_to_strike(f: GeneFragment) -> float:
            dx = min(abs(f.x - strike_x), self.size - abs(f.x - strike_x))
            dy = min(abs(f.y - strike_y), self.size - abs(f.y - strike_y))
            return math.sqrt(dx**2 + dy**2)

        # Sort by distance
        nearby = sorted(self.fragments, key=distance_to_strike)

        # Take the nearest ones within radius
        gathered = []
        for f in nearby:
            if len(gathered) >= num_genes:
                break
            if distance_to_strike(f) <= config.LIGHTNING_GATHER_RADIUS:
                gathered.append(f)

        # Not enough fragments nearby
        if len(gathered) < 3:
            self.lightning_strikes.append(
                LightningStrike(strike_x, strike_y, success=False)
            )
            return

        # Remove gathered fragments from the pool
        for f in gathered:
            self.fragments.remove(f)

        # Try to create a creature
        genes = [f.gene for f in gathered]
        creature = create_creature_from_genes(genes, strike_x, strike_y)

        if creature is not None:
            # Success! Add the creature
            if len(self.creatures) < config.MAX_CREATURES:
                self.creatures.append(creature)
                self.total_births += 1
                self.successful_lightning += 1
            self.lightning_strikes.append(
                LightningStrike(strike_x, strike_y, success=True)
            )
        else:
            # Failed - unstable or non-viable
            self.lightning_strikes.append(
                LightningStrike(strike_x, strike_y, success=False)
            )

    def _update_creatures(self):
        """Update all creatures for one timestep."""
        # Shuffle order to avoid bias
        random.shuffle(self.creatures)

        new_creatures = []
        dead_creatures = []

        for creature in self.creatures:
            # Skip if already dead
            if creature.energy <= 0:
                dead_creatures.append(creature)
                continue

            # 1. Gather sensory input
            input_vector = self._gather_input(creature)

            # 2. Run the brain
            with torch.no_grad():
                output, new_context = creature.brain(input_vector, creature.context)
                creature.context = new_context

            # 3. Execute actions
            energy_cost = self._execute_actions(creature, output)

            # 4. Pay energy costs
            base_cost = config.BASE_ENERGY_COST * creature.efficiency
            creature.energy -= (base_cost + energy_cost)

            # 5. Age
            creature.age += 1

            # 6. Maybe reproduce
            if creature.can_reproduce() and len(self.creatures) + len(new_creatures) < config.MAX_CREATURES:
                child = creature.reproduce()
                if child is not None:
                    new_creatures.append(child)
                    self.total_births += 1

            # 7. Track position for movement interest scoring
            cid = creature.creature_id
            if cid not in self.position_history:
                self.position_history[cid] = []
            self.position_history[cid].append((creature.x, creature.y))
            # Keep last 30 positions for pattern analysis
            if len(self.position_history[cid]) > 30:
                self.position_history[cid] = self.position_history[cid][-30:]

            # 8. Score movement interest and adjust energy
            interest_score = self._calculate_movement_interest(creature)

            # Interesting movement = survive longer (reduced drain)
            # Boring movement = die faster (increased drain)
            # But FOOD is still needed to actually gain energy for reproduction!
            if interest_score > 0.5:
                # Interesting - small energy bonus (but not enough to replace food)
                creature.energy = min(creature.energy + 0.5, creature.max_energy)
            elif interest_score < 0.2:
                # Boring - extra energy drain
                creature.energy -= 1.5

            # 9. Check for death
            if creature.energy <= 0:
                dead_creatures.append(creature)
            elif random.random() < config.RANDOM_DEATH_CHANCE:
                dead_creatures.append(creature)

        # Remove dead creatures
        for creature in dead_creatures:
            if creature in self.creatures:
                self.creatures.remove(creature)
                self.total_deaths += 1
                # Clean up position history
                if creature.creature_id in self.position_history:
                    del self.position_history[creature.creature_id]

        # Add new creatures
        self.creatures.extend(new_creatures)

    def _gather_input(self, creature: Creature) -> torch.Tensor:
        """Build the input vector for a creature based on its input genes."""
        inputs = []

        # Get input genes in order
        input_genes = creature.genome.get_genes_in_category(GeneCategory.INPUT)

        for gene in input_genes:
            if gene.gene_type == SENSE_FOOD_NEAR:
                # 4 values: food in adjacent tiles (up, down, left, right)
                vals = [
                    1.0 if (creature.x, (creature.y - 1) % self.size) in self.food else 0.0,
                    1.0 if (creature.x, (creature.y + 1) % self.size) in self.food else 0.0,
                    1.0 if ((creature.x - 1) % self.size, creature.y) in self.food else 0.0,
                    1.0 if ((creature.x + 1) % self.size, creature.y) in self.food else 0.0,
                ]
                inputs.extend(vals)

            elif gene.gene_type == SENSE_FOOD_FAR:
                # 8 values: food in 8 surrounding tiles (radius 2)
                directions = [(-2, 0), (2, 0), (0, -2), (0, 2),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]
                vals = []
                for dx, dy in directions:
                    tx = (creature.x + dx) % self.size
                    ty = (creature.y + dy) % self.size
                    vals.append(1.0 if (tx, ty) in self.food else 0.0)
                inputs.extend(vals)

            elif gene.gene_type == SENSE_ENERGY_SELF:
                # 1 value: own energy level (normalized)
                inputs.append(creature.energy / creature.max_energy)

            elif gene.gene_type == SENSE_CREATURES_NEAR:
                # 4 values: creatures in adjacent tiles
                vals = []
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    tx = (creature.x + dx) % self.size
                    ty = (creature.y + dy) % self.size
                    has_creature = any(c.x == tx and c.y == ty for c in self.creatures if c != creature)
                    vals.append(1.0 if has_creature else 0.0)
                inputs.extend(vals)

            elif gene.gene_type == SENSE_SIGNAL:
                # 4 values: signals in adjacent tiles
                vals = []
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    tx = (creature.x + dx) % self.size
                    ty = (creature.y + dy) % self.size
                    signal_strength = self.signals.get((tx, ty), 0.0)
                    vals.append(signal_strength)
                inputs.extend(vals)

        return torch.tensor(inputs, dtype=torch.float32)

    def _execute_actions(self, creature: Creature, output: torch.Tensor) -> float:
        """Execute actions based on network output. Returns energy cost."""
        energy_cost = 0.0
        output_idx = 0

        # Reset signaling state
        creature.is_signaling = False

        # Get output genes in order
        output_genes = creature.genome.get_genes_in_category(GeneCategory.OUTPUT)

        for gene in output_genes:
            if gene.gene_type == MOVE:
                # 2 values: dx, dy (continuous, threshold to movement)
                dx = output[output_idx].item()
                dy = output[output_idx + 1].item()
                output_idx += 2

                # Convert to discrete movement - no noise, pure network output
                move_x = 0
                move_y = 0
                threshold = config.MOVE_THRESHOLD
                if abs(dx) > threshold:
                    move_x = 1 if dx > 0 else -1
                if abs(dy) > threshold:
                    move_y = 1 if dy > 0 else -1

                if move_x != 0 or move_y != 0:
                    creature.x = (creature.x + move_x) % self.size
                    creature.y = (creature.y + move_y) % self.size
                    energy_cost += config.ACTION_ENERGY_COST * creature.efficiency

            elif gene.gene_type == MOVE_FAST:
                # 1 value: toggle for fast movement (applied to last MOVE)
                toggle = output[output_idx].item()
                output_idx += 1

                if toggle > 0.5:
                    # Move again in same direction (double move)
                    # This stacks with normal MOVE
                    energy_cost += config.ACTION_ENERGY_COST * config.MOVE_FAST_MULTIPLIER * creature.efficiency

            elif gene.gene_type == EAT:
                # 1 value: toggle to eat
                toggle = output[output_idx].item()
                output_idx += 1

                if toggle > 0:  # Any positive value triggers eating
                    pos = (creature.x, creature.y)
                    if pos in self.food:
                        self.food.remove(pos)
                        creature.energy = min(creature.energy + config.FOOD_ENERGY, creature.max_energy)
                    energy_cost += config.ACTION_ENERGY_COST * 0.5 * creature.efficiency  # Eating is cheap

            elif gene.gene_type == EMIT_SIGNAL:
                # 1 value: toggle to emit signal
                toggle = output[output_idx].item()
                output_idx += 1

                if toggle > 0.5:
                    creature.is_signaling = True
                    energy_cost += config.ACTION_ENERGY_COST * creature.efficiency

        return energy_cost

    def _calculate_movement_interest(self, creature: Creature) -> float:
        """
        Score how interesting a creature's movement pattern is (0.0 to 1.0).

        High scores for:
        - Changing directions frequently
        - Using multiple different directions
        - Covering more area
        - Circular or varied patterns

        Low scores for:
        - Staying still
        - Moving in straight lines only
        """
        cid = creature.creature_id
        history = self.position_history.get(cid, [])

        # Need enough history to judge
        if len(history) < 10:
            return 0.5  # Neutral score while building history

        recent = history[-20:]  # Analyze last 20 positions

        # Calculate movements
        movements = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            # Handle wrap-around
            if dx > self.size // 2:
                dx -= self.size
            elif dx < -self.size // 2:
                dx += self.size
            if dy > self.size // 2:
                dy -= self.size
            elif dy < -self.size // 2:
                dy += self.size
            movements.append((dx, dy))

        if not movements:
            return 0.0

        # Metric 1: Movement activity (are they moving at all?)
        actual_moves = sum(1 for dx, dy in movements if dx != 0 or dy != 0)
        activity_score = actual_moves / len(movements)

        if activity_score < 0.3:
            return 0.1  # Too stationary

        # Metric 2: Direction diversity (how many unique directions?)
        unique_directions = set()
        for dx, dy in movements:
            if dx != 0 or dy != 0:
                dir_x = 0 if dx == 0 else (1 if dx > 0 else -1)
                dir_y = 0 if dy == 0 else (1 if dy > 0 else -1)
                unique_directions.add((dir_x, dir_y))

        # Max 8 directions possible, score based on variety
        direction_score = len(unique_directions) / 8.0

        # Metric 3: Direction changes (turning frequency)
        direction_changes = 0
        prev_dir = None
        for dx, dy in movements:
            if dx != 0 or dy != 0:
                curr_dir = (1 if dx > 0 else (-1 if dx < 0 else 0),
                           1 if dy > 0 else (-1 if dy < 0 else 0))
                if prev_dir is not None and curr_dir != prev_dir:
                    direction_changes += 1
                prev_dir = curr_dir

        turn_score = min(1.0, direction_changes / 8.0)  # Cap at 8 turns being "max"

        # Metric 4: Area coverage (unique positions visited)
        unique_positions = set(recent)
        coverage_score = min(1.0, len(unique_positions) / 12.0)

        # Metric 5: Detect circular/rotational patterns (bonus!)
        # Calculate "angular momentum" - are they turning consistently in one direction?
        angles = []
        for dx, dy in movements:
            if dx != 0 or dy != 0:
                angles.append(math.atan2(dy, dx))

        rotation_bonus = 0.0
        if len(angles) >= 5:
            # Check for consistent turning
            angle_diffs = []
            for i in range(1, len(angles)):
                diff = angles[i] - angles[i-1]
                # Normalize to [-pi, pi]
                while diff > math.pi: diff -= 2 * math.pi
                while diff < -math.pi: diff += 2 * math.pi
                angle_diffs.append(diff)

            # Consistent rotation = low variance in angle changes but non-zero mean
            if angle_diffs:
                mean_turn = sum(angle_diffs) / len(angle_diffs)
                if abs(mean_turn) > 0.2:  # They're turning consistently
                    rotation_bonus = 0.2

        # Combine scores
        interest = (
            0.2 * activity_score +
            0.25 * direction_score +
            0.25 * turn_score +
            0.2 * coverage_score +
            rotation_bonus
        )

        return min(1.0, interest)

    def _spawn_food(self):
        """Spawn new food randomly."""
        if len(self.food) >= config.MAX_FOOD:
            return

        for _ in range(config.FOOD_SPAWN_RATE):
            if len(self.food) >= config.MAX_FOOD:
                break
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            self.food.add((x, y))

    def get_stats(self) -> dict:
        """Get current simulation statistics."""
        genome_counts = {}
        total_energy = 0
        total_age = 0
        total_params = 0

        for creature in self.creatures:
            sig = creature.genome.short_signature()
            genome_counts[sig] = genome_counts.get(sig, 0) + 1
            total_energy += creature.energy
            total_age += creature.age
            total_params += creature.brain.count_parameters()

        # Most common genome
        most_common = max(genome_counts.items(), key=lambda x: x[1]) if genome_counts else ("none", 0)

        return {
            'step': self.step_count,
            'creatures': len(self.creatures),
            'fragments': len(self.fragments),
            'food': len(self.food),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'total_lightning': self.total_lightning,
            'successful_lightning': self.successful_lightning,
            'avg_energy': total_energy / len(self.creatures) if self.creatures else 0,
            'avg_age': total_age / len(self.creatures) if self.creatures else 0,
            'avg_params': total_params / len(self.creatures) if self.creatures else 0,
            'most_common_genome': most_common,
            'unique_genomes': len(genome_counts),
        }
