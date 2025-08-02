#!/usr/bin/env python3
"""
FPA Projection Debugger

A streamlined diagnostic tool to analyze the FPA projection and centroiding
process for a single PSF file. This script provides detailed, step-by-step
output to pinpoint failures in the pipeline, especially for diffuse PSFs
from newer generations.
"""

import os
import sys
import argparse
import numpy as np
import logging
import cv2
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file

# Configure logging for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*80)
    print(f"| {title.upper():^76} |")
    print("="*80)

def print_dict(d, indent=2):
    """Prints a dictionary with indentation."""
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}:")
            print_dict(value, indent + 2)
        elif isinstance(value, (np.ndarray, list)) and len(value) > 5:
             print(f"{' ' * indent}{key}: {type(value)} of shape {np.shape(value)}")
        else:
            print(f"{' ' * indent}{key}: {value}")

def analyze_fpa_projection(psf_file_path, magnitude=3.0):
    """
    Runs a detailed diagnostic on the FPA projection and centroiding for a single PSF.

    Args:
        psf_file_path (str): The path to the PSF file to debug.
        magnitude (float): The star magnitude to simulate.
    """
    if not os.path.isfile(psf_file_path):
        logger.error(f"PSF file not found: {psf_file_path}")
        return

    print_header(f"FPA Projection Debugger - {os.path.basename(psf_file_path)}")

    # 1. Initialize Pipeline
    print_header("1. Initializing Pipeline")
    pipeline = StarTrackerPipeline(debug=True)
    logger.info("StarTrackerPipeline initialized in debug mode.")

    # 2. Load PSF Data
    print_header("2. Loading PSF Data")
    try:
        metadata, intensity_data = parse_psf_file(psf_file_path, debug=False)
        if intensity_data.size == 0:
            raise ValueError("No intensity data found in PSF file.")
        psf_data = {'metadata': metadata, 'intensity_data': intensity_data, 'file_path': psf_file_path}
        logger.info(f"Successfully loaded PSF data. Shape: {intensity_data.shape}")
        print_dict(metadata)
    except Exception as e:
        logger.error(f"Failed to load or parse PSF file: {e}")
        return

    # 3. Project PSF to FPA Grid
    print_header("3. Projecting PSF to FPA Grid")
    fpa_psf_data = pipeline.project_psf_to_fpa_grid(psf_data, target_pixel_pitch_um=5.5)
    scaling_info = fpa_psf_data['scaling_info']
    logger.info("PSF projection to FPA grid complete.")
    print_dict(scaling_info)

    # 4. Simulate Star Image
    print_header("4. Simulating Star Image on FPA Grid")
    star_simulation = pipeline.simulate_star(magnitude=magnitude)
    photon_count = star_simulation['photon_count']
    projection_results = pipeline.project_star_with_psf(fpa_psf_data, photon_count, num_simulations=1)
    simulated_image = projection_results['simulations'][0]
    logger.info(f"Simulated a magnitude {magnitude} star, producing {photon_count:.1f} photons.")
    print(f"  - Simulated Image Shape: {simulated_image.shape}")
    print(f"  - Simulated Image Stats: Min={np.min(simulated_image):.2f}, Max={np.max(simulated_image):.2f}, Mean={np.mean(simulated_image):.2f}")

    # 5. Run Centroiding Detection with Debug Hook
    print_header("5. Running Centroid Detection (with Debug Hook)")
    true_centroid_fpa = pipeline.calculate_true_psf_centroid(fpa_psf_data)
    
    # Use the modified function call to get debug data
    # Note: We are intentionally not using the adaptive thresholding from the full FPA simulation
    # to see the raw output of the standard centroiding algorithm first.
    results = pipeline.run_monte_carlo_simulation_fpa_projected(
        psf_data,
        magnitude=magnitude,
        num_trials=1,
        return_debug_data=True
    )
    
    centroid_results = results['centroid_results']
    debug_data = results['debug_data']
    
    logger.info("Centroid detection process finished.")

    # 6. Analyze and Print Debug Data
    print_header("6. Centroiding Debug Analysis")

    # Threshold Map Analysis
    print("\n--- Threshold Map ---")
    threshold_map = debug_data['threshold_maps'][0]
    print(f"  - Shape: {threshold_map.shape}")
    print(f"  - Stats: Min={np.min(threshold_map):.2f}, Max={np.max(threshold_map):.2f}, Mean={np.mean(threshold_map):.2f}")

    # Binary Image Analysis
    print("\n--- Binary Image (Image > Threshold) ---")
    binary_image = debug_data['binary_images'][0]
    white_pixels = np.sum(binary_image)
    total_pixels = binary_image.size
    print(f"  - Pixels above threshold: {white_pixels} / {total_pixels} ({white_pixels/total_pixels:.2%})")

    # Connected Components Analysis
    print("\n--- Connected Components ---")
    labels = debug_data['labels'][0]
    stats = debug_data['stats'][0]
    num_labels = stats.shape[0]
    print(f"  - Number of potential regions found (including background): {num_labels}")
    if num_labels > 1:
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            print(f"    - Region {i}: Area={area} pixels, BBox=[{left},{top},{width},{height}]")
    else:
        print("    - No regions found.")

    # Valid Region Analysis
    print("\n--- Valid Region Filtering ---")
    print(f"  - Detection parameters: min_pixels=3, max_pixels=100")
    if not debug_data['valid_regions'] or not debug_data['valid_regions'][0]:
        print("  - Result: No regions met the size criteria.")
        valid_regions = []
    else:
        valid_regions = debug_data['valid_regions'][0]
        print(f"  - Result: Found {len(valid_regions)} valid region(s): {valid_regions}")

    # Final Results
    print("\n--- Final Centroiding Results ---")
    print_dict(centroid_results)
    if centroid_results['successful_detections'] == 0:
        print("\nCONCLUSION: Centroiding FAILED. Review the steps above to identify the point of failure.")
    else:
        print("\nCONCLUSION: Centroiding SUCCEEDED.")


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Run a detailed FPA projection and centroiding diagnostic on a single PSF file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "psf_file",
        type=str,
        help="Path to the PSF file to be debugged (e.g., 'PSF_sims/Gen_3/0_deg.txt')."
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=3.0,
        help="Star magnitude to simulate (default: 3.0)."
    )
    args = parser.parse_args()

    analyze_fpa_projection(args.psf_file, args.magnitude)

if __name__ == "__main__":
    main()