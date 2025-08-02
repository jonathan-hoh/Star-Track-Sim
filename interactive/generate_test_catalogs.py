#!/usr/bin/env python3
"""
Generate Standardized Synthetic Star Catalogs for Testing

This script uses the enhanced SyntheticCatalogBuilder to create a suite of
test catalogs with varying complexity. These catalogs are saved to disk
and can be used for repeatable, rigorous testing of the BAST matching algorithm.

Usage:
    python3 tools/interactive/generate_test_catalogs.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.multi_star.synthetic_catalog import SyntheticCatalogBuilder

# --- Configuration ---
OUTPUT_DIR = Path("data/catalogs")
FOV_DEG = 10.0  # Standard FOV for the camera

# Define the test scenarios
TEST_SCENARIOS = {
    "baseline_5_stars_spread": {
        "num_in_fov": 5,
        "num_out_fov": 10,
        "distribution": "spread",
        "magnitude_range": (2.0, 6.0),
    },
    "challenging_8_stars_clustered": {
        "num_in_fov": 8,
        "num_out_fov": 20,
        "distribution": "clustered",
        "magnitude_range": (3.0, 7.0),
        "cluster_std_dev_deg": 0.5, # Tightly clustered
    },
    "high_clutter_10_stars": {
        "num_in_fov": 10,
        "num_out_fov": 50, # Many red herrings
        "distribution": "spread",
        "magnitude_range": (2.0, 8.0),
    },
    "sparse_4_stars_wide_separation": {
        "num_in_fov": 4,
        "num_out_fov": 5,
        "distribution": "spread",
        "magnitude_range": (2.0, 5.0),
    },
}

def generate_and_save_catalogs():
    """
    Generates and saves all test catalogs defined in the configuration.
    """
    print("=" * 60)
    print("Generating Standardized Test Catalogs...")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving catalogs to: {OUTPUT_DIR.resolve()}")

    builder = SyntheticCatalogBuilder()

    for name, params in TEST_SCENARIOS.items():
        print(f"\nGenerating catalog: {name}")
        
        try:
            catalog = builder.generate_catalog(
                num_in_fov=params["num_in_fov"],
                num_out_fov=params["num_out_fov"],
                fov_deg=FOV_DEG,
                distribution=params.get("distribution", "spread"),
                magnitude_range=params.get("magnitude_range", (2.0, 5.0)),
                cluster_std_dev_deg=params.get("cluster_std_dev_deg", 1.0)
            )

            # The BAST Catalog object is a pandas DataFrame subclass.
            # We can call DataFrame methods on it directly.
            output_file = OUTPUT_DIR / f"{name}.csv"
            catalog.to_csv(output_file, index=False)

            print(f"  - Successfully generated {len(catalog)} stars.")
            print(f"  - Saved to: {output_file}")

        except Exception as e:
            print(f"  - FAILED to generate catalog '{name}': {e}")

    print("\nCatalog generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    generate_and_save_catalogs()