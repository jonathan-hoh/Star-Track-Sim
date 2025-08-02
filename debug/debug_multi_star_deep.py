#!/usr/bin/env python3
"""
Deep debugging script for multi-star simulation issues

This script investigates the root causes of why only 1 star is being detected
in multi-star scenes when 3 should be present.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_synthetic_catalog():
    """Debug the synthetic catalog generation"""
    print("🔍 DEBUGGING SYNTHETIC CATALOG GENERATION")
    print("=" * 50)
    
    try:
        from src.multi_star.synthetic_catalog import SyntheticCatalogBuilder
        
        catalog_builder = SyntheticCatalogBuilder()
        catalog = catalog_builder.create_triangle_catalog(separation_deg=0.5)
        
        print(f"✅ Catalog created successfully")
        print(f"📊 Catalog shape: {catalog.shape}")
        print(f"📋 Catalog columns: {list(catalog.columns)}")
        print(f"📄 Catalog contents:")
        print(catalog)
        
        # Check RA/Dec values
        print(f"\n📍 Star positions:")
        for i, row in catalog.iterrows():
            print(f"  Star {i}: RA={row['RA']:.6f}, Dec={row['DE']:.6f}, Mag={row['Magnitude']}")
        
        return True, catalog
        
    except Exception as e:
        print(f"❌ Catalog generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def debug_scene_generation(catalog):
    """Debug the scene generation process"""
    print("\n🔍 DEBUGGING SCENE GENERATION")
    print("=" * 50)
    
    try:
        from src.core.star_tracker_pipeline import StarTrackerPipeline
        from src.multi_star.multi_star_pipeline import MultiStarPipeline
        
        pipeline = StarTrackerPipeline()
        multi_star_pipeline = MultiStarPipeline(pipeline)
        
        # Generate scene
        scene_data = multi_star_pipeline.scene_generator.generate_scene(catalog)
        
        print(f"✅ Scene generated successfully")
        print(f"📊 Scene data keys: {list(scene_data.keys())}")
        
        if 'stars' in scene_data:
            print(f"⭐ Number of stars in scene: {len(scene_data['stars'])}")
            for i, star in enumerate(scene_data['stars']):
                print(f"  Star {i}: {star}")
                
        # Add RA/Dec to scene data
        for i, star in enumerate(scene_data['stars']):
            catalog_row = catalog.iloc[star['catalog_idx']]
            star['ra'] = catalog_row['RA']
            star['dec'] = catalog_row['DE']
            print(f"  Added RA/Dec to Star {i}: RA={star['ra']:.6f}, Dec={star['dec']:.6f}")
        
        return True, scene_data
        
    except Exception as e:
        print(f"❌ Scene generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def debug_psf_loading():
    """Debug PSF data loading"""
    print("\n🔍 DEBUGGING PSF DATA LOADING")
    print("=" * 50)
    
    try:
        from src.core.star_tracker_pipeline import StarTrackerPipeline
        
        pipeline = StarTrackerPipeline()
        
        # Load PSF data
        psf_data_dict = pipeline.load_psf_data("PSF_sims/Gen_1", "*0_deg*")
        if not psf_data_dict:
            print("❌ No PSF data loaded")
            return False, None
            
        psf_data = list(psf_data_dict.values())[0]
        print(f"✅ PSF data loaded successfully")
        print(f"📊 PSF data keys: {list(psf_data.keys())}")
        
        if 'intensity_data' in psf_data:
            intensity_shape = psf_data['intensity_data'].shape
            max_intensity = np.max(psf_data['intensity_data'])
            print(f"📏 PSF intensity shape: {intensity_shape}")
            print(f"📊 PSF max intensity: {max_intensity}")
            
        return True, psf_data
        
    except Exception as e:
        print(f"❌ PSF loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def debug_radiometry_rendering(scene_data, psf_data):
    """Debug the radiometry rendering process"""
    print("\n🔍 DEBUGGING RADIOMETRY RENDERING")
    print("=" * 50)
    
    try:
        from src.core.star_tracker_pipeline import StarTrackerPipeline
        from src.multi_star.multi_star_pipeline import MultiStarPipeline
        
        pipeline = StarTrackerPipeline()
        multi_star_pipeline = MultiStarPipeline(pipeline)
        
        print(f"🎬 Rendering scene with {len(scene_data['stars'])} stars...")
        
        # Render scene
        rendered_scene = multi_star_pipeline.radiometry.render_scene(scene_data, psf_data)
        
        print(f"✅ Scene rendered successfully")
        
        if 'detector_image' in rendered_scene:
            detector_shape = rendered_scene['detector_image'].shape
            detector_max = np.max(rendered_scene['detector_image'])
            detector_min = np.min(rendered_scene['detector_image'])
            detector_nonzero = np.count_nonzero(rendered_scene['detector_image'])
            
            print(f"📏 Detector image shape: {detector_shape}")
            print(f"📊 Detector image range: [{detector_min:.6f}, {detector_max:.6f}]")
            print(f"📊 Non-zero pixels: {detector_nonzero} / {detector_shape[0] * detector_shape[1]}")
            
            # Check for star positions on detector
            print(f"📍 Star positions on detector:")
            for i, star in enumerate(rendered_scene['stars']):
                if 'detector_position' in star:
                    pos = star['detector_position']
                    # Sample intensity at star position
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < detector_shape[1] and 0 <= y < detector_shape[0]:
                        intensity_at_star = rendered_scene['detector_image'][y, x]
                        print(f"  Star {i}: position=({pos[0]:.2f}, {pos[1]:.2f}), intensity={intensity_at_star:.6f}")
                    else:
                        print(f"  Star {i}: position=({pos[0]:.2f}, {pos[1]:.2f}) - OUT OF BOUNDS")
                        
            # Create a visualization of the detector image
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(rendered_scene['detector_image'], cmap='hot', origin='lower')
            plt.title('Full Detector Image')
            plt.colorbar()
            
            # Plot star positions
            for i, star in enumerate(rendered_scene['stars']):
                if 'detector_position' in star:
                    pos = star['detector_position']
                    plt.plot(pos[0], pos[1], 'cyan', marker='x', markersize=12, markeredgewidth=3, label=f'Star {i+1}')
            plt.legend()
            
            # Zoomed view of center region
            plt.subplot(1, 2, 2)
            center_x, center_y = detector_shape[1]//2, detector_shape[0]//2
            zoom_size = 200
            x_start = max(0, center_x - zoom_size)
            x_end = min(detector_shape[1], center_x + zoom_size)
            y_start = max(0, center_y - zoom_size)
            y_end = min(detector_shape[0], center_y + zoom_size)
            
            zoomed_region = rendered_scene['detector_image'][y_start:y_end, x_start:x_end]
            plt.imshow(zoomed_region, cmap='hot', origin='lower', 
                      extent=[x_start, x_end, y_start, y_end])
            plt.title('Zoomed Center Region')
            plt.colorbar()
            
            # Plot star positions in zoomed view
            for i, star in enumerate(rendered_scene['stars']):
                if 'detector_position' in star:
                    pos = star['detector_position']
                    if x_start <= pos[0] <= x_end and y_start <= pos[1] <= y_end:
                        plt.plot(pos[0], pos[1], 'cyan', marker='x', markersize=12, markeredgewidth=3, label=f'Star {i+1}')
            
            plt.tight_layout()
            plt.savefig('debug_detector_image.png', dpi=150, bbox_inches='tight')
            print(f"💾 Saved detector image debug plot: debug_detector_image.png")
            plt.show()
        
        return True, rendered_scene
        
    except Exception as e:
        print(f"❌ Radiometry rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def debug_star_detection(detector_image):
    """Debug star detection algorithm"""
    print("\n🔍 DEBUGGING STAR DETECTION")
    print("=" * 50)
    
    try:
        from src.multi_star.peak_detection import detect_stars_peak_method
        
        print(f"🎯 Running star detection on image shape: {detector_image.shape}")
        print(f"📊 Image statistics: min={np.min(detector_image):.6f}, max={np.max(detector_image):.6f}, mean={np.mean(detector_image):.6f}")
        
        # Test with different parameters
        print("\n🔬 Testing with original parameters:")
        centroids1 = detect_stars_peak_method(
            detector_image,
            min_intensity_fraction=0.1,
            min_separation=50
        )
        print(f"  Detected centroids: {len(centroids1)}")
        for i, centroid in enumerate(centroids1):
            print(f"    Centroid {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        print("\n🔬 Testing with more sensitive parameters:")
        centroids2 = detect_stars_peak_method(
            detector_image,
            min_intensity_fraction=0.05,
            min_separation=25
        )
        print(f"  Detected centroids: {len(centroids2)}")
        for i, centroid in enumerate(centroids2):
            print(f"    Centroid {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        print("\n🔬 Testing with very sensitive parameters:")
        centroids3 = detect_stars_peak_method(
            detector_image,
            min_intensity_fraction=0.01,
            min_separation=10
        )
        print(f"  Detected centroids: {len(centroids3)}")
        for i, centroid in enumerate(centroids3):
            print(f"    Centroid {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        return True, centroids1
        
    except Exception as e:
        print(f"❌ Star detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run comprehensive debugging of multi-star pipeline"""
    print("🐛 COMPREHENSIVE MULTI-STAR PIPELINE DEBUG")
    print("=" * 60)
    
    # Step 1: Debug catalog generation
    catalog_success, catalog = debug_synthetic_catalog()
    if not catalog_success:
        return 1
    
    # Step 2: Debug scene generation
    scene_success, scene_data = debug_scene_generation(catalog)
    if not scene_success:
        return 1
    
    # Step 3: Debug PSF loading
    psf_success, psf_data = debug_psf_loading()
    if not psf_success:
        return 1
    
    # Step 4: Debug radiometry rendering
    radiometry_success, rendered_scene = debug_radiometry_rendering(scene_data, psf_data)
    if not radiometry_success:
        return 1
    
    # Step 5: Debug star detection
    detection_success, centroids = debug_star_detection(rendered_scene['detector_image'])
    if not detection_success:
        return 1
    
    print("\n✅ DEBUGGING COMPLETED SUCCESSFULLY!")
    print("\n📊 FINAL SUMMARY:")
    print(f"  Catalog stars: {len(catalog)}")
    print(f"  Scene stars: {len(scene_data['stars'])}")
    print(f"  Detector image shape: {rendered_scene['detector_image'].shape}")
    print(f"  Detected centroids: {len(centroids)}")
    print(f"  Detection success rate: {len(centroids)}/{len(scene_data['stars'])} = {len(centroids)/len(scene_data['stars'])*100:.1f}%")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)