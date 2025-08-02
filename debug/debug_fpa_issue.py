#!/usr/bin/env python3
"""
Debug script to investigate FPA projection issues at higher field angles
"""

import os
import numpy as np
import logging
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.core.psf_plot import parse_psf_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_field_angle_fpa_projection(psf_file, magnitude=3.0, num_trials=5):
    """Test FPA projection for a specific field angle"""
    
    print(f"\n{'='*60}")
    print(f"TESTING: {os.path.basename(psf_file)}")
    print(f"{'='*60}")
    
    # Create pipeline
    pipeline = StarTrackerPipeline(debug=False)
    
    try:
        # Load PSF data
        metadata, intensity_data = parse_psf_file(psf_file)
        psf_data = {
            'metadata': metadata,
            'intensity_data': intensity_data,
            'file_path': psf_file
        }
        
        print(f"✓ PSF loaded successfully: {intensity_data.shape}")
        print(f"  Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        if metadata and 'data_spacing' in metadata:
            print(f"  Data spacing: {metadata['data_spacing']} µm/px")
        
        # Test original PSF analysis
        print(f"\n--- ORIGINAL PSF ANALYSIS ---")
        try:
            original_results = pipeline.run_monte_carlo_simulation(
                psf_data,
                magnitude=magnitude,
                num_trials=num_trials,
                threshold_sigma=5.0,
                adaptive_block_size=32
            )
            print(f"✓ Original analysis completed")
            print(f"  Success rate: {original_results.get('success_rate', 0.0):.2%}")
            print(f"  Mean centroid error: {original_results.get('mean_centroid_error_px', float('nan')):.3f} px")
            print(f"  Mean centroid error: {original_results.get('mean_centroid_error_um', float('nan')):.3f} µm")
            print(f"  Mean vector error: {original_results.get('mean_vector_error_arcsec', float('nan')):.1f} arcsec")
            
        except Exception as e:
            print(f"✗ Original analysis failed: {e}")
            return False
        
        # Test FPA projection analysis
        print(f"\n--- FPA PROJECTION ANALYSIS ---")
        try:
            fpa_results = pipeline.run_monte_carlo_simulation_fpa_projected(
                psf_data,
                magnitude=magnitude,
                num_trials=num_trials,
                threshold_sigma=5.0,
                adaptive_block_size=8,
                target_pixel_pitch_um=5.5
            )
            print(f"✓ FPA analysis completed")
            print(f"  Success rate: {fpa_results.get('success_rate', 0.0):.2%}")
            print(f"  Mean centroid error: {fpa_results.get('mean_centroid_error_px', float('nan')):.3f} px")
            print(f"  Mean centroid error: {fpa_results.get('mean_centroid_error_um', float('nan')):.3f} µm")
            print(f"  Mean vector error: {fpa_results.get('mean_vector_error_arcsec', float('nan')):.1f} arcsec")
            
            # Check if any values are NaN or invalid
            invalid_values = []
            for key, value in fpa_results.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    invalid_values.append(f"{key}={value}")
            
            if invalid_values:
                print(f"⚠ Invalid values found: {invalid_values}")
            
            # Check scaling info
            if 'scaling_info' in fpa_results:
                scaling = fpa_results['scaling_info']
                print(f"  Scaling info:")
                for key, value in scaling.items():
                    print(f"    {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"✗ FPA analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ PSF loading failed: {e}")
        return False

def main():
    """Test all PSF files in Gen_1 to identify the problematic ones"""
    
    psf_directory = "PSF_sims/Gen_1"
    psf_files = [
        "0_deg.txt", "1_deg.txt", "2_deg.txt", "4_deg.txt", "5_deg.txt",
        "7_deg.txt", "8_deg.txt", "9_deg.txt", "11_deg.txt", "12_deg.txt", "14_deg.txt"
    ]
    
    print("="*80)
    print("FPA PROJECTION DEBUG TEST")
    print("Testing all Gen_1 PSF files to identify issues")
    print("="*80)
    
    results = {}
    
    for psf_file in psf_files:
        psf_path = os.path.join(psf_directory, psf_file)
        if os.path.exists(psf_path):
            field_angle = psf_file.replace('_deg.txt', '')
            success = test_field_angle_fpa_projection(psf_path, magnitude=4.0, num_trials=10)
            results[field_angle] = success
        else:
            print(f"⚠ PSF file not found: {psf_path}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    successful = [angle for angle, success in results.items() if success]
    failed = [angle for angle, success in results.items() if not success]
    
    print(f"Successful field angles: {successful}")
    print(f"Failed field angles: {failed}")
    
    if failed:
        print(f"\n⚠ FPA projection failed for field angles: {failed}")
        print("These are likely the angles missing from your plot!")
    else:
        print(f"\n✓ All field angles processed successfully")
        print("The issue might be in the plotting or data storage logic.")

if __name__ == "__main__":
    main() 