"""
Visualization - Beautiful blob creatures in the primordial soup.

Features:
  - Cute blob creatures with faces that vary by genes
  - Smooth animations and gentle movement
  - Clean, high-tech dark aesthetic
  - Glowing food particles
  - Atmospheric effects
"""

import pygame
import math
import os
import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from PIL import Image

import config
from world import World, LightningStrike
from genes import GeneCategory, SENSE_FOOD_NEAR, SENSE_FOOD_FAR, SENSE_CREATURES_NEAR, SENSE_SIGNAL
from genes import EMBED_16, EMBED_32, EMBED_64, EXTRA_LAYER, EXTRA_HEAD, LONG_MEMORY


@dataclass
class Particle:
    """A visual particle effect."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    size: float
    particle_type: str


class Visualizer:
    """Renders beautiful blob creatures in the primordial soup."""

    def __init__(self, world: World, record: bool = False):
        pygame.init()
        pygame.display.set_caption("Primordial Soup - Evolving Tiny Transformers")

        self.world = world
        self.record = record
        self.frames: List[Image.Image] = []

        # Calculate dimensions
        self.grid_width = config.GRID_SIZE * config.CELL_SIZE
        self.grid_height = config.GRID_SIZE * config.CELL_SIZE
        self.sidebar_width = config.WINDOW_WIDTH - self.grid_width

        # Create window
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_title = pygame.font.Font(None, 42)

        # Particle system
        self.particles: List[Particle] = []

        # Smooth creature positions (for interpolation)
        self.creature_positions: Dict[int, Tuple[float, float]] = {}
        self.creature_target_positions: Dict[int, Tuple[int, int]] = {}

        # Animation time
        self.time = 0

        # Previous state for detecting events
        self.prev_creature_ids: set = set()
        self.prev_positions: dict = {}

        # Selected creature
        self.selected_creature = None

        # Running state
        self.running = True
        self.paused = False

        # Pre-render background (cover full window height)
        self.background_surface = pygame.Surface((self.grid_width, config.WINDOW_HEIGHT))
        self._draw_background()

    def _draw_background(self):
        """Create a clean dark gradient background."""
        bg_height = config.WINDOW_HEIGHT
        bg_width = self.grid_width
        cy = bg_height / 2
        cx = bg_width / 2
        max_dist = math.sqrt(cx**2 + cy**2)

        for y in range(bg_height):
            for x in range(bg_width):
                # Distance from center (normalized)
                dist = math.sqrt((x - cx)**2 + (y - cy)**2) / max_dist

                # Dark purple-blue gradient
                r = int(12 + 8 * dist)
                g = int(10 + 6 * dist)
                b = int(25 + 15 * dist)

                self.background_surface.set_at((x, y), (r, g, b))

    def _get_creature_color(self, creature) -> Tuple[int, int, int]:
        """Get a soft pastel color based on genome."""
        h = hash(creature.genome.signature())

        # Generate soft pastel colors
        base_hue = (h % 360)

        # Convert HSV to RGB (soft pastels: high value, medium saturation)
        s = 0.4 + (((h >> 8) % 100) / 100) * 0.3  # 0.4-0.7 saturation
        v = 0.7 + (((h >> 16) % 100) / 100) * 0.2  # 0.7-0.9 value

        # HSV to RGB conversion
        c = v * s
        x = c * (1 - abs((base_hue / 60) % 2 - 1))
        m = v - c

        if base_hue < 60:
            r, g, b = c, x, 0
        elif base_hue < 120:
            r, g, b = x, c, 0
        elif base_hue < 180:
            r, g, b = 0, c, x
        elif base_hue < 240:
            r, g, b = 0, x, c
        elif base_hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def _get_creature_features(self, creature) -> dict:
        """Determine creature visual features based on genes."""
        genome = creature.genome
        h = hash(genome.signature())

        features = {
            'num_bumps': 0,
            'num_eyes': 2,
            'eye_style': 'normal',  # normal, sleepy, wide
            'has_appendage': False,
        }

        # More sensor genes = more eyes (but max 3)
        sensor_count = (
            genome.count_gene(SENSE_FOOD_NEAR) +
            genome.count_gene(SENSE_FOOD_FAR) +
            genome.count_gene(SENSE_CREATURES_NEAR) +
            genome.count_gene(SENSE_SIGNAL)
        )
        features['num_eyes'] = min(3, max(1, sensor_count))

        # Architecture genes affect body shape
        if genome.has_gene(EMBED_64) or genome.has_gene(EXTRA_LAYER):
            features['num_bumps'] = 2
        elif genome.has_gene(EMBED_32) or genome.has_gene(EXTRA_HEAD):
            features['num_bumps'] = 1

        # Memory gene gives appendage
        if genome.has_gene(LONG_MEMORY):
            features['has_appendage'] = True

        # Eye style based on hash
        eye_styles = ['normal', 'sleepy', 'wide']
        features['eye_style'] = eye_styles[h % 3]

        return features

    def _draw_blob_creature(self, surface: pygame.Surface, px: float, py: float,
                           creature, size: float):
        """Draw a cute blob creature with face."""
        color = self._get_creature_color(creature)
        features = self._get_creature_features(creature)

        # Darker and lighter versions for shading
        darker = tuple(max(0, int(c * 0.7)) for c in color)
        lighter = tuple(min(255, int(c * 1.3)) for c in color)
        highlight = tuple(min(255, c + 60) for c in color)

        # Gentle pulsing based on time
        pulse = 1.0 + 0.05 * math.sin(self.time * 0.1 + hash(creature.creature_id))
        size = size * pulse

        # Create a surface for this creature with alpha
        blob_size = int(size * 4)
        blob_surf = pygame.Surface((blob_size, blob_size), pygame.SRCALPHA)
        center = blob_size // 2

        # Draw outer glow
        glow_color = (*color, 30)
        for r in range(int(size * 1.5), int(size * 0.8), -2):
            alpha = int(30 * (1 - r / (size * 1.5)))
            pygame.draw.circle(blob_surf, (*color, alpha), (center, center), r)

        # Draw main body - multiple overlapping circles for organic shape
        # Base blob
        pygame.draw.circle(blob_surf, color, (center, center), int(size))

        # Add bumps based on genes
        if features['num_bumps'] >= 1:
            # Top bump
            bump_y = center - int(size * 0.6)
            pygame.draw.circle(blob_surf, color, (center, bump_y), int(size * 0.5))
            pygame.draw.circle(blob_surf, lighter, (center - 2, bump_y - 2), int(size * 0.3))

        if features['num_bumps'] >= 2:
            # Side bump
            bump_x = center + int(size * 0.5)
            pygame.draw.circle(blob_surf, color, (bump_x, center), int(size * 0.4))

        # Appendage (tail-like)
        if features['has_appendage']:
            tail_x = center - int(size * 0.8)
            tail_y = center + int(size * 0.3)
            pygame.draw.circle(blob_surf, darker, (tail_x, tail_y), int(size * 0.35))
            pygame.draw.circle(blob_surf, darker, (tail_x - int(size * 0.3), tail_y + int(size * 0.2)), int(size * 0.2))

        # Inner highlight (3D effect)
        highlight_x = center - int(size * 0.3)
        highlight_y = center - int(size * 0.3)
        pygame.draw.circle(blob_surf, (*highlight, 80), (highlight_x, highlight_y), int(size * 0.4))

        # Draw face
        self._draw_face(blob_surf, center, center, size, features)

        # Blit to main surface
        surface.blit(blob_surf, (int(px - center), int(py - center)))

        # Signaling effect
        if creature.is_signaling:
            ring_pulse = (math.sin(self.time * 0.3) + 1) / 2
            ring_size = int(size * 1.5 + 5 * ring_pulse)
            pygame.draw.circle(surface, (255, 230, 150), (int(px), int(py)), ring_size, 2)

    def _draw_face(self, surface: pygame.Surface, cx: int, cy: int, size: float, features: dict):
        """Draw cute face on blob."""
        eye_style = features['eye_style']
        num_eyes = features['num_eyes']

        # Eye positions
        eye_y = cy - int(size * 0.1)
        eye_spacing = int(size * 0.35)
        eye_size = max(2, int(size * 0.15))

        if num_eyes == 1:
            # Cyclops - one big eye
            eye_positions = [(cx, eye_y)]
            eye_size = int(eye_size * 1.5)
        elif num_eyes == 2:
            eye_positions = [(cx - eye_spacing, eye_y), (cx + eye_spacing, eye_y)]
        else:  # 3 eyes
            eye_positions = [
                (cx - eye_spacing, eye_y),
                (cx, eye_y - int(size * 0.15)),
                (cx + eye_spacing, eye_y)
            ]

        for ex, ey in eye_positions:
            if eye_style == 'sleepy':
                # Sleepy eyes - horizontal lines
                pygame.draw.line(surface, (30, 30, 40),
                               (ex - eye_size, ey), (ex + eye_size, ey), 2)
            elif eye_style == 'wide':
                # Wide eyes - bigger circles
                pygame.draw.circle(surface, (240, 240, 250), (ex, ey), eye_size + 1)
                pygame.draw.circle(surface, (30, 30, 40), (ex, ey), eye_size)
                # Highlight
                pygame.draw.circle(surface, (255, 255, 255), (ex - 1, ey - 1), max(1, eye_size // 3))
            else:
                # Normal eyes - simple dots
                pygame.draw.circle(surface, (30, 30, 40), (ex, ey), eye_size)
                # Tiny highlight
                pygame.draw.circle(surface, (80, 80, 90), (ex - 1, ey - 1), max(1, eye_size // 2))

    def handle_events(self) -> bool:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self._save_screenshot()
                elif event.key == pygame.K_g:
                    self._export_gif()
                elif event.key == pygame.K_f:
                    config.SHOW_FRAGMENTS = not config.SHOW_FRAGMENTS

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_click(event.pos)

        return True

    def _handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click to select creature."""
        x, y = pos
        if x >= self.grid_width:
            return

        # Find closest creature to click
        min_dist = float('inf')
        self.selected_creature = None

        for creature in self.world.creatures:
            cid = creature.creature_id
            if cid in self.creature_positions:
                px, py = self.creature_positions[cid]
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_dist and dist < 30:
                    min_dist = dist
                    self.selected_creature = creature

    def update(self):
        """Update visual state with smooth animations."""
        self.time += 1

        # Update smooth creature positions (interpolation for fluid movement)
        for creature in self.world.creatures:
            cid = creature.creature_id
            target_x = creature.x * config.CELL_SIZE + config.CELL_SIZE // 2
            target_y = creature.y * config.CELL_SIZE + config.CELL_SIZE // 2

            if cid not in self.creature_positions:
                # New creature - start at target
                self.creature_positions[cid] = (float(target_x), float(target_y))
            else:
                # Smooth interpolation toward target
                curr_x, curr_y = self.creature_positions[cid]
                lerp_speed = 0.15  # Slower = smoother
                new_x = curr_x + (target_x - curr_x) * lerp_speed
                new_y = curr_y + (target_y - curr_y) * lerp_speed
                self.creature_positions[cid] = (new_x, new_y)

        # Clean up positions for dead creatures
        alive_ids = {c.creature_id for c in self.world.creatures}
        dead_ids = set(self.creature_positions.keys()) - alive_ids

        for cid in dead_ids:
            if cid in self.creature_positions:
                # Spawn death particles
                px, py = self.creature_positions[cid]
                self._spawn_death_particles(px, py)
                del self.creature_positions[cid]

        # Spawn birth particles for new creatures
        for creature in self.world.creatures:
            if creature.creature_id not in self.prev_creature_ids:
                px, py = self.creature_positions.get(creature.creature_id,
                    (creature.x * config.CELL_SIZE, creature.y * config.CELL_SIZE))
                self._spawn_birth_particles(px, py, self._get_creature_color(creature))

        # Update particles
        new_particles = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.05  # Gentle gravity
            p.vx *= 0.98  # Drag
            p.vy *= 0.98
            p.life -= 0.02
            if p.life > 0:
                new_particles.append(p)
        self.particles = new_particles

        self.prev_creature_ids = alive_ids

    def _spawn_death_particles(self, x: float, y: float):
        """Spawn soft particles when creature dies."""
        for _ in range(8):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 0.5,
                color=(200, 150, 180),
                life=1.0,
                size=random.uniform(2, 5),
                particle_type='death'
            ))

    def _spawn_birth_particles(self, x: float, y: float, color: Tuple[int, int, int]):
        """Spawn sparkle particles when creature is born."""
        # Bright ring burst effect
        for i in range(16):
            angle = (i / 16) * 2 * math.pi
            speed = random.uniform(2, 4)
            bright_color = tuple(min(255, c + 80) for c in color)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=bright_color,
                life=1.2,
                size=random.uniform(3, 6),
                particle_type='birth'
            ))
        # Extra sparkles
        for _ in range(8):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=(255, 255, 200),  # Golden sparkle
                life=1.5,
                size=random.uniform(2, 4),
                particle_type='birth'
            ))

    def render(self):
        """Render one frame."""
        self.update()

        # Draw background
        self.screen.blit(self.background_surface, (0, 0))

        # Draw food as glowing orbs
        self._draw_food()

        # Draw gene fragments if enabled
        if config.SHOW_FRAGMENTS:
            self._draw_fragments()

        # Draw creatures
        self._draw_creatures()

        # Draw lightning effects
        self._draw_lightning()

        # Draw particles
        self._draw_particles()

        # Draw selection highlight
        if self.selected_creature and self.selected_creature in self.world.creatures:
            self._draw_selection()
        else:
            self.selected_creature = None

        # Draw sidebar
        self._draw_sidebar()

        # Update display
        pygame.display.flip()

        # Record frame
        if self.record:
            self._capture_frame()

        self.clock.tick(config.FPS)

    def _draw_food(self):
        """Draw food as soft glowing orbs."""
        for (x, y) in self.world.food:
            px = x * config.CELL_SIZE + config.CELL_SIZE // 2
            py = y * config.CELL_SIZE + config.CELL_SIZE // 2

            # Gentle pulse
            pulse = 1.0 + 0.2 * math.sin(self.time * 0.08 + x * 0.5 + y * 0.3)

            # Outer glow
            glow_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            for r in range(8, 2, -1):
                alpha = int(40 * (1 - r / 8))
                pygame.draw.circle(glow_surf, (100, 255, 150, alpha), (10, 10), int(r * pulse))
            self.screen.blit(glow_surf, (px - 10, py - 10))

            # Core
            pygame.draw.circle(self.screen, (150, 255, 180), (px, py), int(3 * pulse))
            pygame.draw.circle(self.screen, (200, 255, 220), (px, py), int(2 * pulse))

    def _draw_fragments(self):
        """Draw gene fragments as tiny floating particles."""
        for fragment in self.world.fragments:
            px = int(fragment.x * config.CELL_SIZE)
            py = int(fragment.y * config.CELL_SIZE)

            base_color = config.GENE_COLORS.get(
                fragment.gene.category.name.lower(),
                (150, 150, 150)
            )

            # Twinkle
            twinkle = (math.sin(self.time * 0.15 + px * 0.1 + py * 0.1) + 1) / 2
            alpha = int(30 + 40 * twinkle)

            surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*base_color, alpha), (3, 3), 2)
            self.screen.blit(surf, (px - 3, py - 3))

    def _draw_creatures(self):
        """Draw all creatures as cute blobs."""
        # Sort by y position for proper layering
        sorted_creatures = sorted(self.world.creatures,
                                  key=lambda c: self.creature_positions.get(c.creature_id, (0, 0))[1])

        for creature in sorted_creatures:
            cid = creature.creature_id
            if cid not in self.creature_positions:
                continue

            px, py = self.creature_positions[cid]

            # Size based on energy
            energy_ratio = creature.energy / creature.max_energy
            size = max(8, int(12 + 8 * energy_ratio))

            self._draw_blob_creature(self.screen, px, py, creature, size)

    def _draw_lightning(self):
        """Draw lightning effects."""
        for strike in self.world.lightning_strikes:
            px = strike.x * config.CELL_SIZE + config.CELL_SIZE // 2
            py = strike.y * config.CELL_SIZE + config.CELL_SIZE // 2

            if strike.age > 15:
                continue

            # Flash effect
            if strike.age < 3:
                flash_alpha = int(150 * (1 - strike.age / 3))
                flash_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
                pygame.draw.circle(flash_surf, (255, 255, 200, flash_alpha), (30, 30), 30)
                self.screen.blit(flash_surf, (px - 30, py - 30))

            # Lightning bolt symbol
            if strike.age < 10:
                alpha = int(255 * (1 - strike.age / 10))
                color = (255, 255, 150) if strike.success else (150, 150, 200)

                # Simple bolt shape
                bolt_points = [
                    (px - 3, py - 12),
                    (px + 2, py - 2),
                    (px - 1, py - 2),
                    (px + 4, py + 12),
                    (px - 1, py + 2),
                    (px + 2, py + 2),
                ]
                pygame.draw.polygon(self.screen, color, bolt_points)

            # Expanding ring
            ring_radius = strike.age * 4 + 10
            ring_alpha = int(80 * (1 - strike.age / 15))
            if ring_alpha > 0:
                ring_color = (255, 255, 200) if strike.success else (150, 150, 180)
                pygame.draw.circle(self.screen, ring_color, (px, py), ring_radius, 1)

    def _draw_particles(self):
        """Draw all particles."""
        for p in self.particles:
            alpha = max(0, min(255, int(180 * p.life)))
            size = int(p.size * p.life)
            if size < 1:
                continue

            surf = pygame.Surface((size * 2 + 4, size * 2 + 4), pygame.SRCALPHA)
            # Ensure color values are valid integers
            r, g, b = int(p.color[0]), int(p.color[1]), int(p.color[2])
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            # Soft glow
            pygame.draw.circle(surf, (r, g, b, alpha // 3), (size + 2, size + 2), size + 2)
            pygame.draw.circle(surf, (r, g, b, alpha), (size + 2, size + 2), size)
            self.screen.blit(surf, (int(p.x - size - 2), int(p.y - size - 2)))

    def _draw_selection(self):
        """Draw selection highlight around selected creature."""
        if not self.selected_creature:
            return

        cid = self.selected_creature.creature_id
        if cid not in self.creature_positions:
            return

        px, py = self.creature_positions[cid]

        # Animated ring
        pulse = (math.sin(self.time * 0.1) + 1) / 2
        radius = 25 + int(5 * pulse)

        # Multiple rings
        for i in range(2):
            r = radius + i * 5
            alpha = 200 - i * 80
            pygame.draw.circle(self.screen, (255, 255, 255), (int(px), int(py)), r, 2)

    def _draw_sidebar(self):
        """Draw the statistics sidebar."""
        sidebar_x = self.grid_width

        # Clean dark background
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.sidebar_width, config.WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (18, 16, 28), sidebar_rect)

        # Accent line
        pygame.draw.line(self.screen, (40, 38, 60), (sidebar_x, 0), (sidebar_x, config.WINDOW_HEIGHT), 2)

        stats = self.world.get_stats()

        # Title
        y = 25
        title_surf = self.font_title.render("PRIMORDIAL", True, (180, 170, 220))
        title_rect = title_surf.get_rect(centerx=sidebar_x + self.sidebar_width // 2, top=y)
        self.screen.blit(title_surf, title_rect)
        y += 35

        title_surf2 = self.font_medium.render("SOUP", True, (140, 130, 180))
        title_rect2 = title_surf2.get_rect(centerx=sidebar_x + self.sidebar_width // 2, top=y)
        self.screen.blit(title_surf2, title_rect2)
        y += 30

        subtitle = self.font_small.render("Evolving Tiny Transformers", True, (90, 85, 120))
        subtitle_rect = subtitle.get_rect(centerx=sidebar_x + self.sidebar_width // 2, top=y)
        self.screen.blit(subtitle, subtitle_rect)
        y += 30

        # Divider
        pygame.draw.line(self.screen, (40, 38, 60), (sidebar_x + 15, y), (config.WINDOW_WIDTH - 15, y), 1)
        y += 20

        # Stats with clean bars
        stat_items = [
            ("Creatures", stats['creatures'], config.MAX_CREATURES, (140, 180, 255)),
            ("Food", stats['food'], config.MAX_FOOD, (140, 255, 180)),
            ("Fragments", stats['fragments'], 500, (180, 140, 220)),
        ]

        for label, value, max_val, color in stat_items:
            text = self.font_small.render(label, True, (120, 115, 150))
            self.screen.blit(text, (sidebar_x + 20, y))

            val_text = self.font_small.render(str(value), True, color)
            self.screen.blit(val_text, (sidebar_x + self.sidebar_width - 50, y))

            y += 20
            bar_width = self.sidebar_width - 40
            bar_height = 4
            bar_x = sidebar_x + 20

            # Background
            pygame.draw.rect(self.screen, (30, 28, 45), (bar_x, y, bar_width, bar_height), border_radius=2)
            # Fill
            fill_width = int(bar_width * min(1, value / max_val))
            if fill_width > 0:
                pygame.draw.rect(self.screen, color, (bar_x, y, fill_width, bar_height), border_radius=2)
            y += 15

        y += 5

        # More stats
        pygame.draw.line(self.screen, (40, 38, 60), (sidebar_x + 15, y), (config.WINDOW_WIDTH - 15, y), 1)
        y += 15

        more_stats = [
            ("Births", f"{stats['total_births']:,}"),
            ("Deaths", f"{stats['total_deaths']:,}"),
            ("Lightning", f"{stats['successful_lightning']}/{stats['total_lightning']}"),
            ("Genomes", f"{stats['unique_genomes']}"),
            ("Avg Age", f"{stats['avg_age']:.0f}"),
        ]

        for label, value in more_stats:
            text = self.font_small.render(f"{label}:", True, (90, 85, 120))
            self.screen.blit(text, (sidebar_x + 20, y))
            val_text = self.font_small.render(value, True, (160, 155, 190))
            self.screen.blit(val_text, (sidebar_x + 110, y))
            y += 18

        # Selected creature info
        if self.selected_creature and self.selected_creature in self.world.creatures:
            y += 10
            pygame.draw.line(self.screen, (40, 38, 60), (sidebar_x + 15, y), (config.WINDOW_WIDTH - 15, y), 1)
            y += 15

            text = self.font_medium.render("Selected", True, (180, 170, 220))
            self.screen.blit(text, (sidebar_x + 20, y))
            y += 25

            c = self.selected_creature
            info = [
                f"Age: {c.age}",
                f"Energy: {c.energy:.0f}/{c.max_energy:.0f}",
                f"Brain: {c.brain.embed_dim}d {c.brain.num_heads}h",
                f"Params: {c.brain.count_parameters():,}",
            ]

            for line in info:
                text = self.font_small.render(line, True, (140, 135, 170))
                self.screen.blit(text, (sidebar_x + 25, y))
                y += 18

        # Controls at bottom
        y = config.WINDOW_HEIGHT - 80
        pygame.draw.line(self.screen, (40, 38, 60), (sidebar_x + 15, y), (config.WINDOW_WIDTH - 15, y), 1)
        y += 15

        controls = [
            ("SPACE", "Pause"),
            ("S", "Screenshot"),
            ("G", "Export GIF"),
        ]

        for key, action in controls:
            key_surf = self.font_small.render(key, True, (120, 140, 200))
            self.screen.blit(key_surf, (sidebar_x + 20, y))
            action_surf = self.font_small.render(action, True, (90, 85, 120))
            self.screen.blit(action_surf, (sidebar_x + 70, y))
            y += 18

    def _capture_frame(self):
        """Capture current frame for export."""
        frame_data = pygame.image.tostring(self.screen, 'RGB')
        frame = Image.frombytes('RGB', (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), frame_data)
        self.frames.append(frame)

        if len(self.frames) > 500:
            self.frames = self.frames[-500:]

    def _save_screenshot(self):
        """Save screenshot."""
        os.makedirs("output", exist_ok=True)
        filename = f"output/screenshot_{self.world.step_count}.png"
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved: {filename}")

    def _export_gif(self):
        """Export GIF."""
        if not self.frames:
            print("No frames to export!")
            return

        os.makedirs("output", exist_ok=True)
        filename = f"output/primordial_soup_{self.world.step_count}.gif"

        self.frames[0].save(
            filename,
            save_all=True,
            append_images=self.frames[1:],
            duration=50,
            loop=0
        )
        print(f"GIF exported: {filename} ({len(self.frames)} frames)")

    def close(self):
        """Clean up."""
        pygame.quit()
