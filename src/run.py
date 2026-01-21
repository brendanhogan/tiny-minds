#!/usr/bin/env python3
"""
Primordial Soup Simulation
==========================

Run this file to start the simulation!

Controls:
  SPACE - Pause/Resume
  S     - Save screenshot
  G     - Export GIF of recent frames
  Click - Select a creature to see its details
  ESC   - Quit

The simulation will:
  1. Start with gene fragments floating in the soup
  2. Lightning will strike and fuse genes into creatures
  3. Creatures that are "dynamically stable" will survive
  4. They'll try to find food, eat, and reproduce
  5. Over time, evolution will favor better genomes

Watch as life emerges from chaos!
"""

import sys
import os

# Add the src directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from world import World
from visualize import Visualizer


def print_banner():
    """Print a nice startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              🧬 PRIMORDIAL SOUP 🧬                        ║
    ║                                                           ║
    ║         Evolving Tiny Minds from Chaos                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("  Starting simulation...")
    print(f"  Grid size: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"  Initial fragments: {config.INITIAL_FRAGMENTS}")
    print(f"  Lightning chance: {config.LIGHTNING_CHANCE}")
    print()


def run_simulation(record: bool = False, max_steps: int = None):
    """
    Run the primordial soup simulation.

    Args:
        record: If True, record frames for video export
        max_steps: Stop after this many steps (None = run forever)
    """
    print_banner()

    # Create the world
    print("  Creating world...")
    world = World()

    # Create visualizer
    print("  Starting visualization...")
    print()
    print("  Controls:")
    print("    SPACE - Pause/Resume")
    print("    S     - Screenshot")
    print("    G     - Export GIF")
    print("    Click - Select creature")
    print("    ESC   - Quit")
    print()

    viz = Visualizer(world, record=record)

    # Main loop
    try:
        while viz.running:
            # Handle input
            if not viz.handle_events():
                break

            # Update simulation (if not paused)
            if not viz.paused:
                world.step()

                # Print stats periodically
                if world.step_count % config.LOG_INTERVAL == 0:
                    stats = world.get_stats()
                    print(f"  Step {stats['step']:5d} | "
                          f"Creatures: {stats['creatures']:3d} | "
                          f"Food: {stats['food']:3d} | "
                          f"Births: {stats['total_births']:4d} | "
                          f"Lightning: {stats['successful_lightning']}/{stats['total_lightning']}")

            # Render
            viz.render()

            # Check max steps
            if max_steps and world.step_count >= max_steps:
                print(f"\n  Reached {max_steps} steps, stopping.")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    # Cleanup
    print("\n  Shutting down...")

    # Export video if recording
    if record and viz.frames:
        print("  Exporting recording...")
        viz.export_video()

    viz.close()

    # Print final stats
    print("\n  Final Statistics:")
    stats = world.get_stats()
    print(f"    Total steps: {stats['step']}")
    print(f"    Final population: {stats['creatures']}")
    print(f"    Total births: {stats['total_births']}")
    print(f"    Total deaths: {stats['total_deaths']}")
    print(f"    Successful lightning: {stats['successful_lightning']}/{stats['total_lightning']}")
    print(f"    Unique genomes seen: {stats['unique_genomes']}")
    print()


def run_headless(steps: int = 1000):
    """
    Run simulation without visualization (for testing/benchmarking).

    Args:
        steps: Number of steps to run
    """
    print_banner()
    print("  Running headless (no visualization)...")
    print()

    world = World()

    for step in range(steps):
        world.step()

        if step % 100 == 0:
            stats = world.get_stats()
            print(f"  Step {stats['step']:5d} | "
                  f"Creatures: {stats['creatures']:3d} | "
                  f"Births: {stats['total_births']:4d}")

    print("\n  Done!")
    stats = world.get_stats()
    print(f"  Final population: {stats['creatures']}")
    print(f"  Total births: {stats['total_births']}")


def main():
    """Entry point for the simulation."""
    import argparse

    parser = argparse.ArgumentParser(description="Primordial Soup Simulation")
    parser.add_argument("--record", action="store_true",
                        help="Record frames for video export")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--steps", type=int, default=None,
                        help="Maximum steps to run")

    args = parser.parse_args()

    if args.headless:
        run_headless(args.steps or 1000)
    else:
        run_simulation(record=args.record, max_steps=args.steps)


if __name__ == "__main__":
    main()
