#!/usr/bin/env python3
"""
Diagnostic script to examine FPA projection in detail
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def diagnose_fpa_projection(psf_file_path, magnitude=3.0):
    """
    Diagnose FPA projection step by step
    """
    # Create pipeline
    pipeline = StarTrackerPipeline(debug=True)
    
    # Load PSF data
    metadata, intensity_data = parse_psf_file(psf_file_path)
    psf_data = {
        'metadata': metadata,
        'intensity_data': intensity_data,
        'file_path': psf_file_path
    }
    
    print("="*60)
    print("STEP 1: ORIGINAL PSF ANALYSIS")
    print("="*60)
    
    # Show original PSF stats
    print(f"Original PSF shape: {intensity_data.shape}")
    print(f"Original PSF min/max: {np.min(intensity_data):.3e} / {np.max(intensity_data):.3e}")
    print(f"Original PSF total intensity: {np.sum(intensity_data):.3e}")
    
    # Calculate original centroid
    true_centroid_orig = pipeline.calculate_true_psf_centroid(psf_data, use_zemax_offsets=False)
    print(f"Original PSF centroid: {true_centroid_orig}")
    
    print("\n" + "="*60)
    print("STEP 2: FPA PROJECTION")
    print("="*60)
    
    # Project to FPA
    fpa_psf_data = pipeline.project_psf_to_fpa_grid(psf_data, target_pixel_pitch_um=5.5)
    fpa_intensity = fpa_psf_data['intensity_data']
    
    print(f"FPA PSF shape: {fpa_intensity.shape}")
    print(f"FPA PSF min/max: {np.min(fpa_intensity):.3e} / {np.max(fpa_intensity):.3e}")
    print(f"FPA PSF total intensity: {np.sum(fpa_intensity):.3e}")
    
    # Calculate FPA centroid
    true_centroid_fpa = pipeline.calculate_true_psf_centroid(fpa_psf_data, use_zemax_offsets=False)
    print(f"FPA PSF centroid: {true_centroid_fpa}")
    
    print("\n" + "="*60)
    print("STEP 3: SINGLE STAR SIMULATION")
    print("="*60)
    
    # Simulate single star on FPA grid
    star_simulation = pipeline.simulate_star(magnitude=magnitude)
    photon_count = star_simulation['photon_count']
    print(f"Star photon count: {photon_count:.1f}")
    
    # Project star using FPA PSF (single trial)
    projection_results = pipeline.project_star_with_psf(
        fpa_psf_data, photon_count, num_simulations=1
    )
    
    # Get the simulated image
    simulated_image = projection_results['simulations'][0]
    print(f"Simulated image shape: {simulated_image.shape}")
    print(f"Simulated image min/max: {np.min(simulated_image):.1f} / {np.max(simulated_image):.1f}")
    print(f"Simulated image total: {np.sum(simulated_image):.1f}")
    
    # Find pixels above background
    background_level = np.median(simulated_image)
    signal_pixels = simulated_image > background_level + 3*np.std(simulated_image)
    num_signal_pixels = np.sum(signal_pixels)
    print(f"Background level: {background_level:.1f}")
    print(f"Signal pixels (>bg+3σ): {num_signal_pixels}")
    
    # Show where signal pixels are
    if num_signal_pixels > 0:
        signal_coords = np.where(signal_pixels)
        print(f"Signal pixel coordinates: {list(zip(signal_coords[0], signal_coords[1]))}")
        print(f"Signal pixel values: {simulated_image[signal_pixels]}")
    
    print("\n" + "="*60)
    print("STEP 4: VISUALIZATION")
    print("="*60)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original PSF
    im1 = ax1.imshow(intensity_data, cmap='viridis', origin='lower')
    ax1.plot(true_centroid_orig[0], true_centroid_orig[1], 'r+', markersize=10, markeredgewidth=2)
    ax1.set_title('Original PSF (128x128, 0.5µm/px)')
    plt.colorbar(im1, ax=ax1)
    
    # FPA-projected PSF
    im2 = ax2.imshow(fpa_intensity, cmap='viridis', origin='lower')
    ax2.plot(true_centroid_fpa[0], true_centroid_fpa[1], 'r+', markersize=10, markeredgewidth=2)
    ax2.set_title('FPA PSF (11x11, 5.5µm/px)')
    plt.colorbar(im2, ax=ax2)
    
    # Simulated image
    im3 = ax3.imshow(simulated_image, cmap='viridis', origin='lower')
    ax3.plot(true_centroid_fpa[0], true_centroid_fpa[1], 'r+', markersize=10, markeredgewidth=2)
    if num_signal_pixels > 0:
        ax3.plot(signal_coords[1], signal_coords[0], 'go', markersize=4, alpha=0.7)
    ax3.set_title(f'Simulated Image ({photon_count:.0f} photons)')
    plt.colorbar(im3, ax=ax3)
    
    # Intensity profile
    center_row = int(true_centroid_fpa[1])
    center_col = int(true_centroid_fpa[0])
    if 0 <= center_row < simulated_image.shape[0]:
        row_profile = simulated_image[center_row, :]
        ax4.plot(row_profile, 'b-', label=f'Row {center_row}')
    if 0 <= center_col < simulated_image.shape[1]:
        col_profile = simulated_image[:, center_col]
        ax4.plot(col_profile, 'r-', label=f'Col {center_col}')
    ax4.axhline(background_level, color='k', linestyle='--', alpha=0.5, label='Background')
    ax4.axhline(background_level + 3*np.std(simulated_image), color='k', linestyle=':', alpha=0.5, label='Bg+3σ')
    ax4.set_title('Intensity Profiles')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"fpa_diagnostic_{psf_file_path.split('/')[-1].replace('.txt', '')}.png", dpi=150)
    plt.show()
    
    return {
        'original_psf': psf_data,
        'fpa_psf': fpa_psf_data, 
        'simulated_image': simulated_image,
        'signal_pixels': num_signal_pixels,
        'photon_count': photon_count
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fpa_diagnostic.py <psf_file>")
        sys.exit(1)
    
    psf_file = sys.argv[1]
    results = diagnose_fpa_projection(psf_file, magnitude=3.0)
    
    print(f"\nDiagnostic complete. Signal pixels detected: {results['signal_pixels']}") 