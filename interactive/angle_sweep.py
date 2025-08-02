#!/usr/bin/env python3
"""
angle_sweep.py - Analyze centroiding performance across different field angles
Compares original PSF simulation grid vs FPA-projected (CMV4000) performance
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pathlib import Path
import re
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file, get_psf_generation_label
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_field_angle_from_filename(filename):
    """Extract field angle from PSF filename (e.g., '12.5_deg.txt')"""
    # Look for patterns like "12.5_deg.txt" or "0_deg.txt"
    match = re.search(r"([\d\.]+)_deg\.txt", filename)
    if match:
        try:
            angle_str = match.group(1)
            return float(angle_str)
        except ValueError:
            logger.warning(f"Could not convert extracted angle '{angle_str}' to float from filename {filename}")
            return None
    return None

def run_field_angle_sweep(psf_directory=".", magnitude=3.0, num_trials=50, output_dir="field_angle_comparison", psf_gen_num=None, debug_parser=False, use_descriptive_labels=True, apply_pixel_center_offset=False):
    """
    Run angle sweep comparing original PSF vs FPA projection
    
    Args:
        psf_directory: Directory containing PSF files
        magnitude: Star magnitude to simulate
        num_trials: Number of Monte Carlo trials per field angle
        output_dir: Base output directory for results
        psf_gen_num: The PSF generation number (e.g., 1, 2)
        debug_parser: Boolean, if True, enables debug prints in parse_psf_file
        use_descriptive_labels: If True, use descriptive labels for supported generations
        apply_pixel_center_offset: If True, shift PSF center from pixel corner to pixel center
    """
    # Create pipeline
    pipeline = StarTrackerPipeline(debug=False)
    
    # Find PSF files - now looking for any .txt file
    psf_pattern = os.path.join(psf_directory, "*.txt")
    psf_files = glob.glob(psf_pattern)
    
    if not psf_files:
        logger.error(f"No PSF files (*.txt) found in directory: {psf_directory}")
        return None
    
    logger.info(f"Found {len(psf_files)} PSF files in {psf_directory}")
    logger.info(f"Running angle sweep with magnitude {magnitude}, {num_trials} trials per angle")
    
    # Create generation-specific output directory
    if psf_gen_num is not None:
        gen_specific_output_dir = os.path.join(output_dir, f"Gen_{psf_gen_num}_Results")
    else:
        gen_specific_output_dir = os.path.join(output_dir, "Unknown_Gen_Results") # Fallback
    
    os.makedirs(gen_specific_output_dir, exist_ok=True)
    logger.info(f"Saving results to: {gen_specific_output_dir}")
    
    # Storage for results
    results = {
        'field_angles': [],
        'original': {
            'centroid_errors_px': [],
            'centroid_stds_px': [],
            'centroid_errors_um': [],
            'centroid_stds_um': [],
            'vector_errors_arcsec': [],
            'vector_stds_arcsec': []
        },
        'fpa': {
            'centroid_errors_px': [],
            'centroid_stds_px': [],
            'centroid_errors_um': [],
            'centroid_stds_um': [],
            'vector_errors_arcsec': [],
            'vector_stds_arcsec': []
        }
    }
    
    # Process each PSF file
    for psf_file in psf_files:
        filename = os.path.basename(psf_file)
        
        # Extract field angle
        field_angle = extract_field_angle_from_filename(filename)
        if field_angle is None:
            logger.warning(f"Could not extract field angle from {filename}, skipping")
            continue
        
        logger.info(f"Processing field angle {field_angle}° ({filename})")
        
        try:
            # Load PSF data
            metadata, intensity_data = parse_psf_file(psf_file, debug=debug_parser)
            
            
            psf_data = {
                'metadata': metadata,
                'intensity_data': intensity_data,
                'file_path': psf_file
            }
            
            # === ORIGINAL PSF ANALYSIS ===
            logger.info(f"  Running original PSF analysis...")
            original_results = pipeline.run_monte_carlo_simulation(
                psf_data,
                magnitude=magnitude,
                num_trials=num_trials,
                threshold_sigma=4.5,
                adaptive_block_size=32
            )
            
            logger.info(f"DEBUG: original_results for {filename}:\n{original_results}") # DEBUG PRINT
            
            # === FPA PROJECTION ANALYSIS ===
            logger.info(f"  Running FPA projection analysis...")
            fpa_results = pipeline.run_monte_carlo_simulation_fpa_projected(
                psf_data,
                magnitude=magnitude,
                num_trials=num_trials,
                threshold_sigma=4.5,
                adaptive_block_size=8,
                target_pixel_pitch_um=5.5,
                apply_coordinate_shift=apply_pixel_center_offset
            )
            
            logger.info(f"DEBUG: fpa_results for {filename}:\n{fpa_results}") # DEBUG PRINT
            
            # Store results
            results['field_angles'].append(field_angle)
            
            # Original PSF results
            results['original']['centroid_errors_px'].append(original_results['mean_centroid_error_px'])
            results['original']['centroid_stds_px'].append(original_results['std_centroid_error_px'])
            # Use dynamically calculated micron errors
            results['original']['centroid_errors_um'].append(original_results['mean_centroid_error_um'])
            results['original']['centroid_stds_um'].append(original_results['std_centroid_error_um'])
            results['original']['vector_errors_arcsec'].append(original_results['mean_vector_error_arcsec'])
            results['original']['vector_stds_arcsec'].append(original_results['std_vector_error_arcsec'])
            
            # FPA results
            results['fpa']['centroid_errors_px'].append(fpa_results['mean_centroid_error_px'])
            results['fpa']['centroid_stds_px'].append(fpa_results['std_centroid_error_px'])
            results['fpa']['centroid_errors_um'].append(fpa_results['mean_centroid_error_um'])
            results['fpa']['centroid_stds_um'].append(fpa_results['std_centroid_error_um'])
            results['fpa']['vector_errors_arcsec'].append(fpa_results['mean_vector_error_arcsec'])
            results['fpa']['vector_stds_arcsec'].append(fpa_results['std_vector_error_arcsec'])
            
            logger.info(f"    Original: {original_results['mean_centroid_error_px']:.3f} PSF px, {original_results['mean_centroid_error_um']:.2f} µm, {original_results['mean_vector_error_arcsec']:.1f} arcsec")
            logger.info(f"    FPA:      {fpa_results['mean_centroid_error_px']:.3f} FPA px ({fpa_results['mean_centroid_error_um']:.2f} µm), {fpa_results['mean_vector_error_arcsec']:.1f} arcsec")
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing {filename} in angle_sweep: {str(e)}", exc_info=True) # Added exc_info=True for full traceback
            # Optionally, store NaNs or skip appending for this angle if critical data is missing
            # For now, just logging and continuing to see if it affects other angles.
            continue
    
    if not results['field_angles']:
        logger.error("No valid results obtained!")
        return None
    
    # Sort results by field angle
    sorted_indices = np.argsort(results['field_angles'])
    for key in results.keys():
        if isinstance(results[key], list):
            results[key] = [results[key][i] for i in sorted_indices]
        elif isinstance(results[key], dict):
            for subkey in results[key].keys():
                results[key][subkey] = [results[key][subkey][i] for i in sorted_indices]
    
    # Generate plots
    # Try to get the data spacing from the first PSF file's metadata for plot labeling
    first_psf_file_path = psf_files[0] if psf_files else None
    original_psf_data_spacing_for_plot = "N/A"
    if first_psf_file_path:
        try:
            first_metadata, _ = parse_psf_file(first_psf_file_path, debug=debug_parser)
            if first_metadata and 'data_spacing' in first_metadata and first_metadata['data_spacing'] is not None:
                original_psf_data_spacing_for_plot = f"{first_metadata['data_spacing']:.3f}"
        except Exception as e:
            logger.warning(f"Could not read metadata from first PSF file for plot label: {e}")

    create_comparison_plots(results, gen_specific_output_dir, magnitude, original_psf_data_spacing_for_plot, psf_gen_num, use_descriptive_labels)
    
    # Save CSV data
    save_results_csv(results, gen_specific_output_dir, magnitude)
    
    logger.info(f"Angle sweep completed! Results saved to {gen_specific_output_dir}")
    return results

def create_comparison_plots(results, output_dir, magnitude, original_psf_spacing_label="N/A", psf_gen_num=None, use_descriptive_labels=True):
    """Create comparison plots for original PSF vs FPA projection"""
    
    field_angles = results['field_angles']
    
    # Create figure with subplots - changed to 1x3
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6)) # Adjusted figsize for 1x3
    
    # Colors for consistency
    orig_color = 'blue'
    fpa_color = 'red'
    
    # Plot 1: Centroid Error in Metric Units (µm)
    orig_errors_um = results['original']['centroid_errors_um']
    orig_stds_um = results['original']['centroid_stds_um']
    fpa_errors_um = results['fpa']['centroid_errors_um']
    fpa_stds_um = results['fpa']['centroid_stds_um']
    
    ax1.errorbar(field_angles, orig_errors_um, yerr=orig_stds_um, 
                fmt='o-', color=orig_color, label=f'Original PSF ({original_psf_spacing_label}µm/px)', capsize=3)
    ax1.errorbar(field_angles, fpa_errors_um, yerr=fpa_stds_um, 
                fmt='s-', color=fpa_color, label='FPA Projected (5.5µm/px)', capsize=3)
    ax1.set_xlabel('Field Angle (degrees)')
    ax1.set_ylabel('Centroid Error (µm)')
    ax1.set_title('Centroiding Performance Comparison (Metric Units)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Centroid Error in Pixel Units
    orig_errors_px = results['original']['centroid_errors_px']
    orig_stds_px = results['original']['centroid_stds_px']
    fpa_errors_px = results['fpa']['centroid_errors_px']
    fpa_stds_px = results['fpa']['centroid_stds_px']
    
    ax2.errorbar(field_angles, orig_errors_px, yerr=orig_stds_px, 
                fmt='o-', color=orig_color, label='Original PSF pixels', capsize=3)
    ax2.errorbar(field_angles, fpa_errors_px, yerr=fpa_stds_px, 
                fmt='s-', color=fpa_color, label='FPA pixels', capsize=3)
    ax2.set_xlabel('Field Angle (degrees)')
    ax2.set_ylabel('Centroid Error (pixels in respective coordinate systems)')
    ax2.set_title('Centroiding Performance Comparison (Pixel Units)\nPSF pixels vs FPA pixels')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bearing Vector Error
    orig_vector_errors = results['original']['vector_errors_arcsec']
    orig_vector_stds = results['original']['vector_stds_arcsec']
    fpa_vector_errors = results['fpa']['vector_errors_arcsec']
    fpa_vector_stds = results['fpa']['vector_stds_arcsec']
    
    ax3.errorbar(field_angles, orig_vector_errors, yerr=orig_vector_stds, 
                fmt='o-', color=orig_color, label='Original PSF', capsize=3)
    ax3.errorbar(field_angles, fpa_vector_errors, yerr=fpa_vector_stds, 
                fmt='s-', color=fpa_color, label='FPA Projected', capsize=3)
    ax3.set_xlabel('Field Angle (degrees)')
    ax3.set_ylabel('Bearing Vector Error (arcsec)')
    ax3.set_title('Bearing Vector Performance Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Create descriptive title
    if psf_gen_num is not None:
        gen_label = get_psf_generation_label(psf_gen_num, use_descriptive_labels)
        title = f'Field Angle Performance Comparison - {gen_label} (Magnitude {magnitude})'
    else:
        title = f'Field Angle Performance Comparison (Magnitude {magnitude})'
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(output_dir, f"field_angle_comparison_mag_{magnitude:.1f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Saved comparison plots to {plot_path}")

def save_results_csv(results, output_dir, magnitude):
    """Save results to CSV file"""
    
    # Prepare data for CSV
    csv_data = {
        'Field_Angle_deg': results['field_angles'],
        'Original_Centroid_Error_px': results['original']['centroid_errors_px'],
        'Original_Centroid_Std_px': results['original']['centroid_stds_px'],
        'Original_Centroid_Error_um': results['original']['centroid_errors_um'],
        'Original_Centroid_Std_um': results['original']['centroid_stds_um'],
        'Original_Vector_Error_arcsec': results['original']['vector_errors_arcsec'],
        'Original_Vector_Std_arcsec': results['original']['vector_stds_arcsec'],
        'FPA_Centroid_Error_px': results['fpa']['centroid_errors_px'],
        'FPA_Centroid_Std_px': results['fpa']['centroid_stds_px'],
        'FPA_Centroid_Error_um': results['fpa']['centroid_errors_um'],
        'FPA_Centroid_Std_um': results['fpa']['centroid_stds_um'],
        'FPA_Vector_Error_arcsec': results['fpa']['vector_errors_arcsec'],
        'FPA_Vector_Std_arcsec': results['fpa']['vector_stds_arcsec']
    }
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f"field_angle_comparison_mag_{magnitude:.1f}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved results CSV to {csv_path}")

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Field angle performance comparison: Original PSF vs FPA projection")
    # parser.add_argument("--psf-dir", default=".", help="Directory containing PSF files (default: current directory)")
    parser.add_argument("--magnitude", type=float, default=3.0, help="Star magnitude to simulate (default: 3.0)")
    parser.add_argument("--trials", type=int, default=50, help="Number of Monte Carlo trials (default: 50)")
    parser.add_argument("--output", default="angle_sweep_outputs", help="Base output directory (default: angle_sweep_outputs)")
    parser.add_argument("--debug-parser", action="store_true", help="Enable detailed debug logging for PSF file parsing.")
    parser.add_argument("--simple-labels", action="store_true", help="Use simple 'Generation X' labels instead of descriptive labels")
    parser.add_argument("--pixel-center-offset", action="store_true", help="Apply 0.5 pixel offset to simulate worst-case centroiding scenario (PSF center on pixel center rather than corner)")
    
    args = parser.parse_args()

    # Get PSF generation from user
    psf_directory = None
    psf_gen_num_for_path = None # To store the successfully parsed number
    while True:
        try:
            psf_gen_num_str = input("Which generation of PSF files? (Enter number, e.g., 1, 2): ")
            if not psf_gen_num_str.strip(): # Handle empty input
                print("Input cannot be empty. Please enter a number.")
                continue
            psf_gen_num_for_path = int(psf_gen_num_str)
            if psf_gen_num_for_path <= 0:
                print("Please enter a positive integer for the generation number.")
                continue
            
            psf_directory = os.path.join("PSF_sims", f"Gen_{psf_gen_num_for_path}")

            if not os.path.isdir(psf_directory):
                logger.error(f"PSF directory not found: {psf_directory}")
                logger.error(f"Please ensure the 'PSF_sims/Gen_{psf_gen_num_for_path}' folder structure exists and contains PSF files.")
                print(f"Error: PSF directory '{psf_directory}' not found. Please check the path and try again.")
                # Allow user to re-enter if directory not found, or exit if they enter nothing
                retry_input = input("Press Enter to try again, or type 'exit' to quit: ")
                if retry_input.lower() == 'exit':
                    sys.exit(1)
                psf_directory = None # Reset to re-trigger prompt
                continue 
            break  # Valid input and directory found
        except ValueError:
            print("Invalid input. Please enter a whole number (e.g., 1, 2).")
            psf_gen_num_for_path = None # Reset on error
        except Exception as e: # Catch other potential errors during input/path construction
            logger.error(f"An unexpected error occurred while getting PSF generation: {e}")
            print("An unexpected error occurred. Please try again.")
            psf_gen_num_for_path = None # Reset on error
            # Allow user to re-enter or exit
            retry_input = input("Press Enter to try again, or type 'exit' to quit: ")
            if retry_input.lower() == 'exit':
                sys.exit(1)
            psf_directory = None # Reset to re-trigger prompt
            continue

    if psf_directory is None or psf_gen_num_for_path is None: # Should not happen if loop breaks correctly, but as a safeguard
        logger.error("Failed to determine PSF directory or generation number. Exiting.")
        sys.exit(1)
    
    # Get star magnitude from user, with default
    star_magnitude = args.magnitude # Default from argparse
    try:
        mag_input_str = input(f"Enter star magnitude to simulate (default: {star_magnitude}): ")
        if mag_input_str.strip(): # If user provided input
            star_magnitude = float(mag_input_str)
            if star_magnitude < -26.74: # Apparent magnitude of the Sun
                logger.warning("Extremely bright magnitude entered, may lead to saturation or unrealistic results.")
            elif star_magnitude > 30: # Very faint limit
                logger.warning("Extremely faint magnitude entered, may lead to no detections.")
        else:
            logger.info(f"No magnitude input, using default: {star_magnitude}")
    except ValueError:
        logger.warning(f"Invalid magnitude input. Using default: {star_magnitude}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting magnitude input: {e}. Using default: {args.magnitude}")
        star_magnitude = args.magnitude # Fallback to initial default from args

    logger.info("="*80)
    logger.info("FIELD ANGLE PERFORMANCE COMPARISON")
    logger.info("Original PSF Simulation Grid vs FPA-Projected (CMV4000)")
    logger.info("="*80)
    logger.info(f"Using PSF directory: {psf_directory}")
    logger.info(f"Star magnitude: {star_magnitude}")
    logger.info(f"Monte Carlo trials: {args.trials}")
    logger.info(f"Base output directory: {args.output}")
    logger.info(f"Pixel center offset: {args.pixel_center_offset} {'(worst-case scenario)' if args.pixel_center_offset else '(best-case scenario)'}")
    
    # Run the analysis
    results = run_field_angle_sweep(
        psf_directory=psf_directory,
        magnitude=star_magnitude,
        num_trials=args.trials,
        output_dir=args.output,
        psf_gen_num=psf_gen_num_for_path,
        debug_parser=args.debug_parser,
        use_descriptive_labels=not args.simple_labels,
        apply_pixel_center_offset=args.pixel_center_offset
    )
    
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Calculate summary statistics
        orig_centroid_mean = np.mean(results['original']['centroid_errors_um'])
        fpa_centroid_mean = np.mean(results['fpa']['centroid_errors_um'])
        orig_vector_mean = np.mean([x for x in results['original']['vector_errors_arcsec'] if not np.isnan(x)])
        fpa_vector_mean = np.mean([x for x in results['fpa']['vector_errors_arcsec'] if not np.isnan(x)])
        
        print(f"Average Centroid Performance:")
        print(f"  Original PSF:  {orig_centroid_mean:.3f} µm")
        print(f"  FPA Projected: {fpa_centroid_mean:.3f} µm")
        print(f"  Difference:    {(fpa_centroid_mean - orig_centroid_mean):+.3f} µm ({( (fpa_centroid_mean - orig_centroid_mean) / orig_centroid_mean * 100) if orig_centroid_mean else float('inf'):+.1f}%)")
        
        print(f"\nAverage Bearing Vector Performance:")
        print(f"  Original PSF:  {orig_vector_mean:.2f} arcsec")
        print(f"  FPA Projected: {fpa_vector_mean:.2f} arcsec")
        print(f"  Difference:    {(fpa_vector_mean - orig_vector_mean):+.2f} arcsec ({( (fpa_vector_mean - orig_vector_mean) / orig_vector_mean * 100) if orig_vector_mean else float('inf'):+.1f}%)")
        
        print(f"\nResults saved to: {args.output}")
        print("="*80)
    else:
        logger.error("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()