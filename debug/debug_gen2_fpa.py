#!/usr/bin/env python3
"""
Debug script to investigate Gen 2 PSF FPA projection issues
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def debug_gen2_projection(psf_file):
    """Debug Gen 2 PSF projection to understand scaling issues"""
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING GEN 2 PSF: {os.path.basename(psf_file)}")
    print(f"{'='*60}")
    
    # Create pipeline
    pipeline = StarTrackerPipeline(debug=False)
    
    # Load PSF data
    metadata, intensity_data = parse_psf_file(psf_file)
    psf_data = {
        'metadata': metadata,
        'intensity_data': intensity_data,
        'file_path': psf_file
    }
    
    print(f"📊 PSF PARAMETERS:")
    print(f"  Grid size: {intensity_data.shape}")
    print(f"  Data spacing: {metadata.get('data_spacing')} µm/pixel")
    print(f"  Data area: {metadata.get('data_area')}")
    print(f"  Physical grid size: {intensity_data.shape[0] * float(metadata.get('data_spacing')):.2f} x {intensity_data.shape[1] * float(metadata.get('data_spacing')):.2f} µm")
    
    # Calculate scaling parameters
    psf_pixel_spacing = float(metadata.get('data_spacing'))  # 0.232 µm
    target_fpa_pitch = 5.5  # µm
    scale_factor = target_fpa_pitch / psf_pixel_spacing
    scale_factor_int = int(np.round(scale_factor))
    
    print(f"\n🔧 SCALING CALCULATION:")
    print(f"  PSF spacing: {psf_pixel_spacing} µm/pixel")
    print(f"  CMV4000 spacing: {target_fpa_pitch} µm/pixel")
    print(f"  Scale factor: {scale_factor:.2f}")
    print(f"  Scale factor (rounded): {scale_factor_int}")
    print(f"  Expected FPA grid: {intensity_data.shape[0] // scale_factor_int} x {intensity_data.shape[1] // scale_factor_int}")
    
    # Try the actual FPA projection
    print(f"\n🔄 RUNNING FPA PROJECTION:")
    try:
        fpa_psf_data = pipeline.project_psf_to_fpa_grid(psf_data, target_pixel_pitch_um=5.5)
        fpa_intensity = fpa_psf_data['intensity_data']
        
        print(f"  ✅ Projection successful!")
        print(f"  FPA grid size: {fpa_intensity.shape}")
        print(f"  FPA min/max intensity: {np.min(fpa_intensity):.3e} / {np.max(fpa_intensity):.3e}")
        print(f"  FPA total intensity: {np.sum(fpa_intensity):.3e}")
        
        # Show scaling info
        scaling_info = fpa_psf_data['scaling_info']
        print(f"\n📏 SCALING INFO:")
        for key, value in scaling_info.items():
            print(f"    {key}: {value}")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original PSF
        im1 = ax1.imshow(intensity_data, cmap='viridis', origin='lower')
        ax1.set_title(f'Original PSF\n{intensity_data.shape}, {psf_pixel_spacing} µm/px')
        plt.colorbar(im1, ax=ax1)
        
        # Zoomed original PSF (if large enough)
        center = np.array(intensity_data.shape) // 2
        crop_size = min(16, intensity_data.shape[0] // 2)
        cropped = intensity_data[center[0]-crop_size:center[0]+crop_size, 
                                center[1]-crop_size:center[1]+crop_size]
        im2 = ax2.imshow(cropped, cmap='viridis', origin='lower')
        ax2.set_title(f'Original PSF (cropped)\n{cropped.shape}')
        plt.colorbar(im2, ax=ax2)
        
        # FPA projected PSF
        if fpa_intensity.size > 1:  # Only plot if there's more than 1 pixel
            im3 = ax3.imshow(fpa_intensity, cmap='viridis', origin='lower')
            ax3.set_title(f'FPA Projected\n{fpa_intensity.shape}, {target_fpa_pitch} µm/px')
            plt.colorbar(im3, ax=ax3)
        else:
            ax3.text(0.5, 0.5, f'FPA grid too small!\n{fpa_intensity.shape}', 
                    ha='center', va='center', transform=ax3.transAxes, 
                    fontsize=12, color='red')
            ax3.set_title('FPA Projected (FAILED)')
        
        # Intensity profiles
        if intensity_data.shape[0] > 1:
            center_row = intensity_data.shape[0] // 2
            ax4.plot(intensity_data[center_row, :], 'b-', label='Original PSF')
        
        if fpa_intensity.size > 1 and fpa_intensity.shape[0] > 1:
            # Scale the FPA profile to match original for comparison
            fpa_center_row = fpa_intensity.shape[0] // 2
            fpa_profile = fpa_intensity[fpa_center_row, :]
            x_fpa = np.linspace(0, intensity_data.shape[1]-1, len(fpa_profile))
            ax4.plot(x_fpa, fpa_profile * np.max(intensity_data[center_row, :]) / np.max(fpa_profile), 
                    'r-', label='FPA Projected (scaled)')
        
        ax4.set_title('Intensity Profiles')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Gen 2 PSF Projection Analysis: {os.path.basename(psf_file)}', 
                     fontsize=14, y=0.98)
        
        plt.savefig(f"gen2_debug_{os.path.basename(psf_file).replace('.txt', '')}.png", 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Projection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test Gen 2 PSF files"""
    
    gen2_files = [
        "PSF_sims/Gen_2/0_deg.txt",
        "PSF_sims/Gen_2/10_deg.txt", 
        "PSF_sims/Gen_2/14_deg.txt"
    ]
    
    print("="*80)
    print("GEN 2 PSF PROJECTION DEBUG")
    print("="*80)
    
    for psf_file in gen2_files:
        if os.path.exists(psf_file):
            success = debug_gen2_projection(psf_file)
            print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: {psf_file}")
        else:
            print(f"⚠️  File not found: {psf_file}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 