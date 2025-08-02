#!/usr/bin/env python3
"""
Debug script to test the improved centroiding algorithm with FPA projection
"""

import sys
import os

# Add project root to Python path so we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file, discover_psf_generations_and_angles, get_psf_generation_label, PSF_GENERATION_LABELS
import logging
import argparse
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def debug_single_psf_comparison(psf_file_path, magnitude=3.0, num_trials=10, save_plots=True, show_full_fpa=False, output_dir_for_plots=None, psf_gen_num=None, debug_parser_flag=False, use_descriptive_labels=True, apply_pixel_center_offset=False):
    """
    Debug a single PSF file comparing original vs FPA-projected centroiding
    
    Args:
        psf_file_path: Path to PSF file
        magnitude: Star magnitude to simulate
        num_trials: Number of trials to run
        save_plots: Whether to save debug plots
        show_full_fpa: Whether to show full 2048×2048 CMV4000 detector view
        output_dir_for_plots: Directory to save plots into (if save_plots is True)
        psf_gen_num: PSF generation number
        debug_parser_flag: Boolean, if True, enables debug prints in parse_psf_file
        use_descriptive_labels: If True, use descriptive labels for supported generations
        apply_pixel_center_offset: If True, shift PSF center from pixel corner to pixel center
    """
    # Create pipeline
    pipeline = StarTrackerPipeline(debug=True)
    
    # Load PSF data
    try:
        metadata, intensity_data = parse_psf_file(psf_file_path, debug=debug_parser_flag)
        
        
        psf_data = {
            'metadata': metadata,
            'intensity_data': intensity_data,
            'file_path': psf_file_path
        }
    except Exception as e:
        logger.error(f"Error loading PSF file: {e}")
        return None
    
    print("\n" + "="*80)
    print("ORIGINAL PSF SIMULATION GRID ANALYSIS")
    print("="*80)
    
    # Run original PSF analysis
    results_original = pipeline.run_monte_carlo_simulation(
        psf_data, 
        magnitude=magnitude,
        num_trials=num_trials,
        threshold_sigma=5.0,
        adaptive_block_size=32
    )
    
    print("\n" + "="*80)
    print("FPA-PROJECTED (CMV4000) ANALYSIS")
    print("="*80)
    
    # Run FPA-projected analysis
    results_fpa = pipeline.run_monte_carlo_simulation_fpa_projected(
        psf_data, 
        magnitude=magnitude,
        num_trials=num_trials,
        threshold_sigma=5.0,
        adaptive_block_size=16,  # Smaller block size for smaller FPA grid
        target_pixel_pitch_um=5.5,
        create_full_fpa=show_full_fpa,  # Create full detector view if requested
        fpa_size=(2048, 2048),
        apply_coordinate_shift=apply_pixel_center_offset
    )
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nOriginal PSF Grid Results:")
    # Access data from original_results which now contains 'psf_data' for metadata access if needed
    orig_psf_meta = results_original.get('psf_data', {}).get('metadata', {})
    orig_actual_spacing = orig_psf_meta.get('data_spacing', 'N/A')
    try:
        orig_actual_spacing_str = f"{float(orig_actual_spacing):.3f}" if isinstance(orig_actual_spacing, (float, int)) else str(orig_actual_spacing)
    except ValueError:
        orig_actual_spacing_str = str(orig_actual_spacing)

    print(f"  Grid size: {results_original['projection_results']['simulations'][0].shape} (Spacing: {orig_actual_spacing_str} µm/px)")
    # print(f"  Success rate: {results_original['centroid_results'].get('success_rate', 0.0):.2%}") // Removed success rate
    print(f"  Mean centroid error: {results_original['mean_centroid_error_px']:.3f} ± {results_original.get('std_centroid_error_px', float('nan')):.3f} PSF pixels")
    print(f"  Mean centroid error (µm): {results_original.get('mean_centroid_error_um', float('nan')):.2f} ± {results_original.get('std_centroid_error_um', float('nan')):.2f} µm")
    if not np.isnan(results_original['mean_vector_error_arcsec']):
        print(f"  Mean vector error: {results_original['mean_vector_error_arcsec']:.2f} ± {results_original.get('std_vector_error_arcsec', float('nan')):.2f} arcsec")
    
    print("\nFPA-Projected (CMV4000) Results:")
    print(f"  Grid size: {results_fpa['projection_results']['simulations'][0].shape}")
    print(f"  FPA pixel pitch: {results_fpa['fpa_pixel_pitch_um']:.1f} µm")
    # print(f"  Success rate: {results_fpa['centroid_results'].get('success_rate', 0.0):.2%}") // Removed success rate
    print(f"  Mean centroid error: {results_fpa['mean_centroid_error_px']:.3f} ± {results_fpa.get('std_centroid_error_px', float('nan')):.3f} FPA pixels")
    print(f"  Mean centroid error (µm): {results_fpa['mean_centroid_error_um']:.2f} ± {results_fpa.get('std_centroid_error_um', float('nan')):.2f} µm")
    if not np.isnan(results_fpa['mean_vector_error_arcsec']):
        print(f"  Mean vector error: {results_fpa['mean_vector_error_arcsec']:.2f} ± {results_fpa.get('std_vector_error_arcsec', float('nan')):.2f} arcsec")
    
    # Scale comparison
    scaling_info = results_fpa['scaling_info']
    print(f"\nScaling Information:")
    print(f"  Original PSF: {scaling_info['original_shape']} pixels @ {scaling_info['original_pixel_spacing_um']:.1f} µm/px")
    print(f"  FPA projection: {scaling_info['fpa_shape']} pixels @ {scaling_info['fpa_pixel_spacing_um']:.1f} µm/px")
    print(f"  Scale factor: {scaling_info['scale_factor_int_used']}:1")
    print(f"  Intensity conservation: {scaling_info['intensity_conservation_ratio']:.4f}")
    
    # Full FPA information if created
    if show_full_fpa and scaling_info.get('full_fpa_shape') is not None:
        print(f"\nFull FPA Information:")
        print(f"  Full detector size: {scaling_info['full_fpa_shape']} pixels")
        if scaling_info['fpa_position'] is not None:
            pos = scaling_info['fpa_position']
            print(f"  PSF position on detector: ({pos[0]}, {pos[1]})")
            print(f"  PSF physical position: ({(pos[0]-1024)*5.5/1000:.2f}, {(pos[1]-1024)*5.5/1000:.2f}) mm from center")
            print(f"  PSF area coverage: {(11*11)/(2048*2048)*100:.4f}% of detector")
    
    if save_plots and results_original['projection_results']['simulations'] and results_fpa['projection_results']['simulations']:
        create_comparison_visualization(results_original, results_fpa, psf_file_path, show_full_fpa, output_dir_for_plots, psf_gen_num, use_descriptive_labels, apply_pixel_center_offset)
    
    return {
        'original': results_original,
        'fpa_projected': results_fpa,
        'scaling_info': scaling_info
    }

def create_comparison_visualization(results_original, results_fpa, psf_file_path, show_full_fpa, output_dir_for_plots=None, psf_gen_num=None, use_descriptive_labels=True, apply_pixel_center_offset=False):
    """Create comprehensive comparison visualization"""
    
    # Dynamic labels from metadata
    orig_meta = results_original.get('psf_data', {}).get('metadata', {})
    orig_spacing = orig_meta.get('data_spacing', 'N/A')
    try: # Format spacing if it's a number
        orig_spacing_str = f"{float(orig_spacing):.3f}" if isinstance(orig_spacing, (float, int)) else str(orig_spacing)
    except ValueError:
        orig_spacing_str = str(orig_spacing)

    orig_dims_tuple = orig_meta.get('image_grid_dim')
    orig_dims_str = f"{orig_dims_tuple[1]}×{orig_dims_tuple[0]}" if orig_dims_tuple else "N/AxN/A" # Assuming dim is (width, height), so PSF shape is (height, width)

    fpa_scaling_info = results_fpa.get('scaling_info', {})
    fpa_spacing = fpa_scaling_info.get('fpa_pixel_spacing_um', 5.5)
    fpa_shape_tuple = fpa_scaling_info.get('fpa_shape') # (height, width)
    fpa_dims_str = f"{fpa_shape_tuple[0]}×{fpa_shape_tuple[1]}" if fpa_shape_tuple else "N/AxN/A"

    # Construct a general title prefix with descriptive labels
    if psf_gen_num is not None:
        gen_label = get_psf_generation_label(psf_gen_num, use_descriptive_labels)
        if apply_pixel_center_offset:
            title_prefix = f"{gen_label} - {os.path.basename(psf_file_path)} - Mag {results_original.get('star_simulation',{}).get('magnitude', 'N/A')} - Worst-Case Centroiding (PSF on Pixel Center)"
        else:
            title_prefix = f"{gen_label} - {os.path.basename(psf_file_path)} - Mag {results_original.get('star_simulation',{}).get('magnitude', 'N/A')}"
    else:
        if apply_pixel_center_offset:
            title_prefix = f"{os.path.basename(psf_file_path)} - Mag {results_original.get('star_simulation',{}).get('magnitude', 'N/A')} - Worst-Case Centroiding (PSF on Pixel Center)"
        else:
            title_prefix = f"{os.path.basename(psf_file_path)} - Mag {results_original.get('star_simulation',{}).get('magnitude', 'N/A')}"

    if apply_pixel_center_offset:
        # Create FPA-focused visualization for worst-case centroiding analysis
        if show_full_fpa and results_fpa['fpa_psf_data'].get('full_fpa_intensity') is not None:
            # FPA-focused view with full detector
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(title_prefix, fontsize=14)
            
            # FPA projected PSF (top left)
            ax1 = axes[0, 0]
            fpa_psf = results_fpa['fpa_psf_data']['intensity_data']
            im1 = ax1.imshow(fpa_psf, cmap='hot', origin='lower')
            ax1.set_title(f"FPA Projected PSF (Shifted)\n({fpa_dims_str}, {fpa_spacing:.1f}µm/px)")
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Full FPA detector with PSF (top right)
            ax2 = axes[0, 1]
            full_fpa = results_fpa['fpa_psf_data']['full_fpa_intensity']
            im2 = ax2.imshow(full_fpa, cmap='hot', origin='lower', vmin=0, vmax=np.max(full_fpa)*0.1)
            ax2.set_title('Full CMV4000 Detector\n(2048×2048, 5.5µm/px)')
            
            # Highlight PSF location
            fpa_position = results_fpa['scaling_info']['fpa_position']
            if fpa_position is not None:
                row_start, col_start = fpa_position
                fpa_height, fpa_width = results_fpa['scaling_info']['fpa_shape']
                rect = patches.Rectangle((col_start, row_start), fpa_width, fpa_height, 
                                       linewidth=2, edgecolor='cyan', facecolor='none')
                ax2.add_patch(rect)
                ax2.text(col_start + fpa_width/2, row_start + fpa_height + 50, 'PSF Location', 
                        ha='center', va='bottom', color='cyan', fontweight='bold')
            
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # FPA simulation with centroids (bottom left)
            ax3 = axes[1, 0]
            example_image_fpa = results_fpa['projection_results']['simulations'][0]
            im3 = ax3.imshow(example_image_fpa, cmap='viridis', origin='lower')
            ax3.set_title('FPA Simulated Star\n(Worst-Case Alignment)')
            
            # Mark centroids
            true_centroid_fpa = results_fpa['centroid_results']['true_center']
            ax3.plot(true_centroid_fpa[0], true_centroid_fpa[1], 'r+', markersize=12, markeredgewidth=2, label='True')
            if results_fpa['centroid_results']['centroids']:
                for i, (x, y) in enumerate(results_fpa['centroid_results']['centroids'][:3]):
                    ax3.plot(x, y, 'g+', markersize=8, markeredgewidth=1.5, 
                            label='Detected' if i == 0 else "")
            ax3.legend()
            plt.colorbar(im3, ax=ax3, shrink=0.8)
            
            # Performance summary (bottom right)
            ax4 = axes[1, 1]
            performance_text = f"""FPA Performance (Worst-Case Scenario):

Centroid Error: {results_fpa['mean_centroid_error_px']:.3f} ± {results_fpa.get('std_centroid_error_px', float('nan')):.3f} FPA px
                ({results_fpa['mean_centroid_error_um']:.2f} ± {results_fpa.get('std_centroid_error_um', float('nan')):.2f} µm)

Vector Error: {results_fpa['mean_vector_error_arcsec']:.2f} ± {results_fpa.get('std_vector_error_arcsec', float('nan')):.2f} arcsec

Coordinate Shift: 2.75 µm (½ pixel)
Detection Rate: {len(results_fpa['centroid_results']['centroids'])}/{results_fpa['projection_results'].get('num_simulations', 'N/A')} trials

PSF Grid: {fpa_dims_str} pixels
Physical Size: {fpa_spacing * results_fpa['scaling_info']['fpa_shape'][0]:.1f} × {fpa_spacing * results_fpa['scaling_info']['fpa_shape'][1]:.1f} µm"""
            
            ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=11)
            ax4.set_title('Worst-Case Performance Summary')
            ax4.axis('off')
            
        else:
            # Standard FPA-focused view (2x2 layout)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(title_prefix, fontsize=14)
            
            # FPA projected PSF (top left)
            ax1 = axes[0, 0]
            fpa_psf = results_fpa['fpa_psf_data']['intensity_data']
            im1 = ax1.imshow(fpa_psf, cmap='hot', origin='lower')
            ax1.set_title(f"FPA Projected PSF (Shifted)\n({fpa_dims_str}, {fpa_spacing:.1f}µm/px)")
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # FPA simulation with centroids (top right)
            ax2 = axes[0, 1]
            example_image_fpa = results_fpa['projection_results']['simulations'][0]
            im2 = ax2.imshow(example_image_fpa, cmap='viridis', origin='lower')
            ax2.set_title('FPA Simulated Star\n(Worst-Case Alignment)')
            
            # Mark centroids
            true_centroid_fpa = results_fpa['centroid_results']['true_center']
            ax2.plot(true_centroid_fpa[0], true_centroid_fpa[1], 'r+', markersize=12, markeredgewidth=2, label='True center')
            if results_fpa['centroid_results']['centroids']:
                for x, y in results_fpa['centroid_results']['centroids'][:5]:
                    ax2.plot(x, y, 'g+', markersize=8, markeredgewidth=1.5, label='Detected' if x == results_fpa['centroid_results']['centroids'][0][0] else "")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.colorbar(im2, ax=ax2, label='Intensity')
            
            # Error histogram (bottom left)
            ax3 = axes[1, 0]
            if results_fpa['centroid_results']['centroid_errors']:
                # Convert errors to microns for this plot
                errors_fpa_um = [e * fpa_spacing for e in results_fpa['centroid_results']['centroid_errors']]
                mean_fpa_um = results_fpa['mean_centroid_error_um']
                
                ax3.hist(errors_fpa_um, bins=10, alpha=0.7, color='red', label='FPA (µm)')
                if not np.isnan(mean_fpa_um):
                    ax3.axvline(mean_fpa_um, color='red', linestyle='--', label=f'Mean: {mean_fpa_um:.2f} µm')

            ax3.set_xlabel('Centroid Error (µm)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Centroid Error Distribution\n(Worst-Case Scenario)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Performance summary (bottom right)
            ax4 = axes[1, 1]
            performance_text = f"""FPA Performance (Worst-Case Scenario):

Centroid Error: {results_fpa['mean_centroid_error_px']:.3f} ± {results_fpa.get('std_centroid_error_px', float('nan')):.3f} FPA px
                ({results_fpa['mean_centroid_error_um']:.2f} ± {results_fpa.get('std_centroid_error_um', float('nan')):.2f} µm)

Vector Error: {results_fpa['mean_vector_error_arcsec']:.2f} ± {results_fpa.get('std_vector_error_arcsec', float('nan')):.2f} arcsec

Coordinate Shift: 2.75 µm (½ pixel)

PSF Grid: {fpa_dims_str} pixels
Physical Size: {fpa_spacing * results_fpa['scaling_info']['fpa_shape'][0]:.1f} × {fpa_spacing * results_fpa['scaling_info']['fpa_shape'][1]:.1f} µm

"""
#Detection Rate: {len(results_fpa['centroid_results']['centroids'])}/{results_fpa['projection_results'].get('num_simulations', 'N/A')} trials
            
            ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
            ax4.set_title('Performance Summary')
            ax4.axis('off')
            
    elif show_full_fpa and results_fpa['fpa_psf_data'].get('full_fpa_intensity') is not None:
        # Extended visualization with full FPA
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(title_prefix, fontsize=14)
        
        # === Top Row: PSF Comparisons ===
        
        # Original PSF
        ax1 = axes[0, 0]
        orig_psf = results_original['projection_results']['normalized_psf']
        im1 = ax1.imshow(orig_psf, cmap='hot', origin='lower')
        ax1.set_title(f"Original PSF\n({orig_dims_str}, {orig_spacing_str}µm/px)")
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # FPA projected PSF (e.g. 11x11)
        ax2 = axes[0, 1]
        fpa_psf = results_fpa['fpa_psf_data']['intensity_data']
        im2 = ax2.imshow(fpa_psf, cmap='hot', origin='lower')
        ax2.set_title(f"FPA Projected PSF\n({fpa_dims_str}, {fpa_spacing:.1f}µm/px)")
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Full FPA detector with PSF
        ax3 = axes[0, 2]
        full_fpa = results_fpa['fpa_psf_data']['full_fpa_intensity']
        im3 = ax3.imshow(full_fpa, cmap='hot', origin='lower', vmin=0, vmax=np.max(full_fpa)*0.1)
        ax3.set_title('Full CMV4000 Detector\n(2048×2048, 5.5µm/px)')
        
        # Highlight PSF location
        fpa_position = results_fpa['scaling_info']['fpa_position']
        if fpa_position is not None:
            row_start, col_start = fpa_position
            fpa_height, fpa_width = results_fpa['scaling_info']['fpa_shape']
            rect = patches.Rectangle((col_start, row_start), fpa_width, fpa_height, 
                                   linewidth=2, edgecolor='cyan', facecolor='none')
            ax3.add_patch(rect)
            ax3.text(col_start + fpa_width/2, row_start + fpa_height + 50, 'PSF Location', 
                    ha='center', va='bottom', color='cyan', fontweight='bold')
        
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # === Bottom Row: Simulated Images ===
        
        # Original simulation
        ax4 = axes[1, 0]
        example_image_orig = results_original['projection_results']['simulations'][0]
        im4 = ax4.imshow(example_image_orig, cmap='viridis', origin='lower')
        ax4.set_title('Original PSF\nSimulated Star')
        
        # Mark centroids
        true_centroid_orig = results_original['centroid_results']['true_center']
        ax4.plot(true_centroid_orig[0], true_centroid_orig[1], 'r+', markersize=12, markeredgewidth=2, label='True')
        if results_original['centroid_results']['centroids']:
            for i, (x, y) in enumerate(results_original['centroid_results']['centroids'][:3]):
                ax4.plot(x, y, 'g+', markersize=8, markeredgewidth=1.5, 
                        label='Detected' if i == 0 else "")
        ax4.legend()
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        # FPA simulation
        ax5 = axes[1, 1]
        example_image_fpa = results_fpa['projection_results']['simulations'][0]
        im5 = ax5.imshow(example_image_fpa, cmap='viridis', origin='lower')
        ax5.set_title('FPA Projected\nSimulated Star')
        
        # Mark centroids
        true_centroid_fpa = results_fpa['centroid_results']['true_center']
        ax5.plot(true_centroid_fpa[0], true_centroid_fpa[1], 'r+', markersize=12, markeredgewidth=2, label='True')
        if results_fpa['centroid_results']['centroids']:
            for i, (x, y) in enumerate(results_fpa['centroid_results']['centroids'][:3]):
                ax5.plot(x, y, 'g+', markersize=8, markeredgewidth=1.5, 
                        label='Detected' if i == 0 else "")
        ax5.legend()
        plt.colorbar(im5, ax=ax5, shrink=0.8)
        
        # Performance comparison
        ax6 = axes[1, 2]
        create_performance_comparison_chart(ax6, results_original, results_fpa)
        
    else:
        # Standard visualization (2×2 layout)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title_prefix, fontsize=14)
        
        # Plot original PSF simulation image
        example_image_orig = results_original['projection_results']['simulations'][0]
        im1 = ax1.imshow(example_image_orig, cmap='viridis', origin='lower')
        ax1.set_title(f'Original PSF Grid ({orig_spacing_str} µm/px)')
        
        # Mark centroids on original
        true_centroid_orig = results_original['centroid_results']['true_center']
        ax1.plot(true_centroid_orig[0], true_centroid_orig[1], 'r+', markersize=12, markeredgewidth=2, label='True center')
        if results_original['centroid_results']['centroids']:
            for x, y in results_original['centroid_results']['centroids'][:5]:
                ax1.plot(x, y, 'g+', markersize=8, markeredgewidth=1.5, label='Detected' if x == results_original['centroid_results']['centroids'][0][0] else "")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # Plot FPA-projected image
        example_image_fpa = results_fpa['projection_results']['simulations'][0]
        im2 = ax2.imshow(example_image_fpa, cmap='viridis', origin='lower')
        ax2.set_title(f'FPA-Projected Grid ({fpa_spacing:.1f} µm/px)')
        
        # Mark centroids on FPA
        true_centroid_fpa = results_fpa['centroid_results']['true_center']
        ax2.plot(true_centroid_fpa[0], true_centroid_fpa[1], 'r+', markersize=12, markeredgewidth=2, label='True center')
        if results_fpa['centroid_results']['centroids']:
            for x, y in results_fpa['centroid_results']['centroids'][:5]:
                ax2.plot(x, y, 'g+', markersize=8, markeredgewidth=1.5, label='Detected' if x == results_fpa['centroid_results']['centroids'][0][0] else "")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Intensity')
        
        # Plot error histograms
        if results_original['centroid_results']['centroid_errors']:
            errors_orig = results_original['centroid_results']['centroid_errors']
            ax3.hist(errors_orig, bins=10, alpha=0.7, color='blue', label='Original PSF pixels')
            ax3.axvline(np.mean(errors_orig), color='blue', linestyle='--', label=f'Mean: {np.mean(errors_orig):.3f} PSF px')
        
        if results_fpa['centroid_results']['centroid_errors']:
            errors_fpa_px = results_fpa['centroid_results']['centroid_errors']
            ax3.hist(errors_fpa_px, bins=10, alpha=0.7, color='red', label='FPA pixels')
            ax3.axvline(np.mean(errors_fpa_px), color='red', linestyle='--', label=f'Mean: {np.mean(errors_fpa_px):.3f} FPA px')
        
        ax3.set_xlabel('Centroid Error (pixels in respective coordinate systems)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Centroid Error Distribution (Pixel Units)\nOriginal PSF pixels vs FPA pixels')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot error histograms in metric units
        if results_original['centroid_results']['centroid_errors']:
            # Robustly determine current_psf_data_spacing for the original PSF errors
            _temp_orig_spacing = None
            if orig_meta: # orig_meta is results_original['psf_data']['metadata']
                _temp_orig_spacing = orig_meta.get('data_spacing')

            if isinstance(_temp_orig_spacing, (float, int)) and _temp_orig_spacing > 0:
                current_orig_psf_data_spacing = float(_temp_orig_spacing)
            else:
                logger.warning(f"Could not determine valid positive data_spacing for original PSF histogram (got '{_temp_orig_spacing}'). Defaulting to 0.5 µm/px.")
                current_orig_psf_data_spacing = 0.5 # Default value
            
            label_for_orig_hist_spacing = f"{current_orig_psf_data_spacing:.3f}" # For the histogram label

            # Use the already calculated mean_centroid_error_um if available
            if 'mean_centroid_error_um' in results_original and not np.isnan(results_original['mean_centroid_error_um']):
                mean_orig_um = results_original['mean_centroid_error_um']
                # For histogram, still need individual errors in um, using the validated spacing
                errors_orig_um = [e * current_orig_psf_data_spacing for e in results_original['centroid_results']['centroid_errors'] if e is not None]
            else: # Fallback (should be less common now)
                errors_orig_um = [e * current_orig_psf_data_spacing for e in results_original['centroid_results']['centroid_errors'] if e is not None]
                mean_orig_um = np.mean(errors_orig_um) if errors_orig_um else float('nan')
            
            ax4.hist(errors_orig_um, bins=10, alpha=0.7, color='blue', label=f'Original PSF ({label_for_orig_hist_spacing} µm/px)')
            if not np.isnan(mean_orig_um):
                ax4.axvline(mean_orig_um, color='blue', linestyle='--', label=f'Mean: {mean_orig_um:.2f} µm')
        
        if results_fpa['centroid_results']['centroid_errors']:
            # FPA errors are already calculated with correct spacing in run_monte_carlo_simulation_fpa_projected
            mean_fpa_um = results_fpa['mean_centroid_error_um']
            std_fpa_um = results_fpa['std_centroid_error_um'] # For potential future use, not directly in histogram data here
            # For histogram, we need individual errors in um, which would be error_px * fpa_spacing
            errors_fpa_um = [e * fpa_spacing for e in results_fpa['centroid_results']['centroid_errors']]

            ax4.hist(errors_fpa_um, bins=10, alpha=0.7, color='red', label=f'FPA-projected ({fpa_spacing:.1f} µm/px)')
            if not np.isnan(mean_fpa_um):
                ax4.axvline(mean_fpa_um, color='red', linestyle='--', label=f'Mean: {mean_fpa_um:.2f} µm')
        
        ax4.set_xlabel('Centroid Error (µm)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Centroid Error Distribution (Metric Units)\nPhysically comparable across coordinate systems')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Construct filename and save path
    base_psf_filename = os.path.basename(psf_file_path)
    base_psf_filename_no_ext = os.path.splitext(base_psf_filename)[0]
    if apply_pixel_center_offset:
        plot_filename = f"fpa_worst_case_{base_psf_filename_no_ext}.png"
    else:
        plot_filename = f"fpa_comparison_{base_psf_filename_no_ext}.png"

    if output_dir_for_plots:
        if not os.path.exists(output_dir_for_plots):
            os.makedirs(output_dir_for_plots, exist_ok=True)
        final_plot_path = os.path.join(output_dir_for_plots, plot_filename)
    else: # Fallback to current directory if no output_dir_for_plots is given
        final_plot_path = plot_filename
        
    plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {final_plot_path}")
    if plt.get_fignums(): # Check if there are any figures to show
        plt.show()
    else:
        plt.close(fig) # Explicitly close if not shown to free memory

def create_performance_comparison_chart(ax, results_original, results_fpa):
    """Create performance comparison bar chart (Centroid µm, Vector arcsec only)"""
    
    metrics = ['Centroid Error (µm)', 'Vector Error (arcsec)'] # Removed Success Rate
    
    # Calculate values - use directly from results dict where available
    orig_centroid_um = results_original.get('mean_centroid_error_um', float('nan'))
    fpa_centroid_um = results_fpa.get('mean_centroid_error_um', float('nan'))
    
    orig_vector = results_original.get('mean_vector_error_arcsec', float('nan'))
    fpa_vector = results_fpa.get('mean_vector_error_arcsec', float('nan'))
    
    # orig_success = results_original.get('success_rate', float('nan')) # Removed
    # fpa_success = results_fpa.get('success_rate', float('nan'))   # Removed
    
    # Create grouped bar chart
    x = np.arange(len(metrics)) # Will be np.arange(2)
    width = 0.35
    
    orig_values = [orig_centroid_um, orig_vector] # Removed orig_success
    fpa_values = [fpa_centroid_um, fpa_vector]   # Removed fpa_success
    
    ax.bar(x - width/2, orig_values, width, label='Original PSF', alpha=0.7, color='blue')
    ax.bar(x + width/2, fpa_values, width, label='FPA Projected', alpha=0.7, color='red')
    
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig_val, fpa_val) in enumerate(zip(orig_values, fpa_values)):
        if not np.isnan(orig_val):
            ax.text(i - width/2, orig_val + orig_val*0.01, f'{orig_val:.2f}', 
                   ha='center', va='bottom', fontweight='bold')
        if not np.isnan(fpa_val):
            ax.text(i + width/2, fpa_val + fpa_val*0.01, f'{fpa_val:.2f}', 
                   ha='center', va='bottom', fontweight='bold')

def main():
    parser = argparse.ArgumentParser(description='Debug FPA projection centroiding algorithm')
    parser.add_argument('--magnitude', type=float, default=4.0, help='Star magnitude (default: 3.0)')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials (default: 10)')
    parser.add_argument('--full-fpa', action='store_true', help='Show full 2048×2048 CMV4000 detector view')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation and saving')
    parser.add_argument('--output', default="debug_centroiding_outputs", help="Base output directory for plots (default: debug_centroiding_outputs)")
    parser.add_argument("--debug-parser", action="store_true", help="Enable detailed debug logging for PSF file parsing.")
    parser.add_argument("--simple-labels", action="store_true", help="Use simple 'Generation X' labels instead of descriptive labels")
    parser.add_argument("--pixel-center-offset", action="store_true", help="Apply 0.5 pixel offset to simulate worst-case centroiding scenario (PSF center on pixel center rather than corner)")

    args = parser.parse_args()

    # Discover available PSFs
    available_psfs = discover_psf_generations_and_angles("data/PSF_sims")
    if not available_psfs:
        logger.error("No PSF generations or files found in 'PSF_sims/'. Please check the directory structure and filenames (e.g., PSF_sims/Gen_1/0.0_deg.txt).")
        sys.exit(1)

    # --- PSF Generation Selection ---
    psf_gen_num = None
    selected_gen_angles = []
    sorted_gen_keys = sorted(available_psfs.keys())

    print("Available PSF Generations:")
    for i, gen_key in enumerate(sorted_gen_keys):
        # Show descriptive label if using descriptive labels, otherwise show simple label
        if not args.simple_labels and gen_key in PSF_GENERATION_LABELS:
            display_label = f"{PSF_GENERATION_LABELS[gen_key]} (Gen {gen_key})"
        else:
            display_label = f"Generation {gen_key}"
        print(f"  {i+1}. {display_label}")
    
    while True:
        try:
            gen_choice_str = input("Select PSF Generation (enter number): ")
            if not gen_choice_str.strip(): continue
            gen_choice_idx = int(gen_choice_str) - 1
            if 0 <= gen_choice_idx < len(sorted_gen_keys):
                psf_gen_num = sorted_gen_keys[gen_choice_idx]
                selected_gen_angles = available_psfs[psf_gen_num]
                logger.info(f"Selected PSF Generation: {psf_gen_num}")
                break
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(sorted_gen_keys)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # --- Field Angle Selection ---
    psf_file_path = None
    selected_angle_str = ""

    print(f"\nAvailable Field Angles for Generation {psf_gen_num}:")
    for i, angle in enumerate(selected_gen_angles):
        print(f"  {i+1}. {angle:.2f} deg")

    while True:
        try:
            angle_choice_str = input(f"Select Field Angle for Gen {psf_gen_num} (enter number): ")
            if not angle_choice_str.strip(): continue
            angle_choice_idx = int(angle_choice_str) - 1
            if 0 <= angle_choice_idx < len(selected_gen_angles):
                selected_angle_float = selected_gen_angles[angle_choice_idx]
                # Format to match filename, e.g., 0.0 or 12.5
                if selected_angle_float == int(selected_angle_float):
                    selected_angle_str = f"{int(selected_angle_float)}"
                else:
                    selected_angle_str = f"{selected_angle_float}"
                
                # Construct the filename
                psf_filename = f"{selected_angle_str}_deg.txt"
                psf_file_path = os.path.join("data", "PSF_sims", f"Gen_{psf_gen_num}", psf_filename)
                
                if not os.path.isfile(psf_file_path):
                    logger.error(f"Constructed PSF file path does not exist: {psf_file_path}. This should not happen if discovery was correct.")
                    print(f"Error: File {psf_file_path} not found. Please report this issue.")
                    sys.exit(1) # Critical error if discovered file is not found
                
                logger.info(f"Selected Field Angle: {selected_angle_float} deg (Filename: {psf_filename})")
                break
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(selected_gen_angles)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Prepare output directory
    gen_specific_plot_dir = os.path.join(args.output, f"Gen_{psf_gen_num}_Results", os.path.splitext(os.path.basename(psf_file_path))[0]) # Subfolder per PSF file
    if not args.no_plots:
        os.makedirs(gen_specific_plot_dir, exist_ok=True)
        logger.info(f"Debug plots will be saved to: {gen_specific_plot_dir}")
    else:
        gen_specific_plot_dir = None # Pass None if not saving plots

    logger.info(f"Debugging PSF file: {psf_file_path}")
    logger.info(f"Star magnitude: {args.magnitude}")
    logger.info(f"Number of trials: {args.trials}")
    logger.info(f"Show full FPA: {args.full_fpa}")
    logger.info(f"Pixel center offset: {args.pixel_center_offset} {'(worst-case scenario)' if args.pixel_center_offset else '(best-case scenario)'}")
    
    results = debug_single_psf_comparison(
        psf_file_path, 
        args.magnitude, 
        args.trials, 
        save_plots=not args.no_plots,
        show_full_fpa=args.full_fpa,
        output_dir_for_plots=gen_specific_plot_dir,
        psf_gen_num=psf_gen_num,
        debug_parser_flag=args.debug_parser,
        use_descriptive_labels=not args.simple_labels,
        apply_pixel_center_offset=args.pixel_center_offset
    )
    
    if results:
        print("\n" + "="*80)
        print("FINAL COMPARISON SUMMARY")
        print("="*80)
        orig = results['original']
        fpa = results['fpa_projected']
        
        # Use the mean_centroid_error_um directly from the results dictionary
        orig_centroid_error_um_val = orig.get('mean_centroid_error_um', float('nan'))

        print(f"\nCentroiding Performance:")
        print(f"  Original PSF grid: {orig['mean_centroid_error_px']:.3f} ± {orig['std_centroid_error_px']:.3f} PSF px ({orig_centroid_error_um_val:.2f} µm)")
        print(f"  FPA-projected:     {fpa['mean_centroid_error_px']:.3f} ± {fpa['std_centroid_error_px']:.3f} FPA px ({fpa['mean_centroid_error_um']:.2f} µm)")
        
        print(f"\nBearing Vector Performance:")
        if not np.isnan(orig['mean_vector_error_arcsec']):
            print(f"  Original PSF grid: {orig['mean_vector_error_arcsec']:.2f} ± {orig['std_vector_error_arcsec']:.2f} arcsec")
        if not np.isnan(fpa['mean_vector_error_arcsec']):
            print(f"  FPA-projected:     {fpa['mean_vector_error_arcsec']:.2f} ± {fpa.get('std_vector_error_arcsec', float('nan')):.2f} arcsec")
        
        # Calculate difference for bearing vector
        if not np.isnan(orig['mean_vector_error_arcsec']) and not np.isnan(fpa['mean_vector_error_arcsec']):
            diff_arcsec = fpa['mean_vector_error_arcsec'] - orig['mean_vector_error_arcsec']
            percent_diff_str = "N/A"
            if orig['mean_vector_error_arcsec'] != 0:
                percent_diff = (diff_arcsec / abs(orig['mean_vector_error_arcsec'])) * 100 # Use abs for percentage calculation base
                percent_diff_str = f"{percent_diff:+.1f}%"
            
            print(f"\nBearing Vector Accuracy Difference (FPA - Original):")
            print(f"  Absolute: {diff_arcsec:+.2f} arcsec")
            print(f"  Relative Change: {percent_diff_str}") # Changed language
        
        # Full FPA summary
        if args.full_fpa and results['scaling_info'].get('fpa_position') is not None:
            pos = results['scaling_info']['fpa_position']
            fpa_shape = results['scaling_info'].get('fpa_shape', (0,0))
            detector_shape = results['scaling_info'].get('full_fpa_shape', (1,1)) # Avoid div by zero
            coverage = (fpa_shape[0] * fpa_shape[1]) / (detector_shape[0] * detector_shape[1]) * 100 if (detector_shape[0]*detector_shape[1] > 0) else 0

            print(f"\nFull FPA Context:")
            print(f"  PSF ({fpa_shape[0]}x{fpa_shape[1]}) covers {coverage:.4f}% of detector area ({detector_shape[0]}x{detector_shape[1]})")
            print(f"  Located at detector position: {pos}")

if __name__ == "__main__":
    main() 