#!/usr/bin/env python3
"""
End-to-end test for multi-star BAST matching simulation.

This script validates the complete pipeline from synthetic catalog creation 
through BAST matching, ensuring coordinate transformations are consistent.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.star_tracker_pipeline import StarTrackerPipeline
from .synthetic_catalog import SyntheticCatalogBuilder
from .multi_star_pipeline import MultiStarPipeline
from .coordinate_validation import validate_coordinate_transformations, log_coordinate_pipeline_details
from ..BAST.match import match

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_triangle_matching_test():
    """
    Run a complete 3-star triangle matching test with coordinate validation.
    """
    logger.info("Starting end-to-end triangle matching test...")
    
    try:
        # Step 1: Create synthetic catalog
        logger.info("Step 1: Creating synthetic 3-star catalog...")
        catalog_builder = SyntheticCatalogBuilder()
        catalog = catalog_builder.create_triangle_catalog(separation_deg=2.0)  # Smaller separation for testing
        
        logger.info(f"Created catalog with {len(catalog)} stars")
        logger.info(f"Catalog triplets: {catalog.num_triplets()}")
        
        # Step 2: Initialize pipeline
        logger.info("Step 2: Initializing star tracker pipeline...")
        pipeline = StarTrackerPipeline()
        multi_star_pipeline = MultiStarPipeline(pipeline)
        
        # Step 3: Load PSF data (Gen_1, 0-degree for simplicity)
        logger.info("Step 3: Loading PSF data...")
        psf_data_dict = pipeline.load_psf_data("data/PSF_sims/Gen_1", "*0_deg*")
        if not psf_data_dict:
            raise ValueError("No PSF files found. Check data/PSF_sims/Gen_1/ directory.")
            
        psf_data = list(psf_data_dict.values())[0]  # Use first (and likely only) PSF
        logger.info(f"Loaded PSF: {psf_data['metadata'].get('field_angle', 'unknown')} degrees")
        
        # Step 4: Generate scene
        logger.info("Step 4: Generating multi-star scene...")
        scene_generator = multi_star_pipeline.scene_generator
        scene_data = scene_generator.generate_scene(catalog)
        
        # Add RA/Dec to scene data for validation
        for i, star in enumerate(scene_data['stars']):
            catalog_row = catalog.iloc[star['catalog_idx']]
            star['ra'] = catalog_row['RA']
            star['dec'] = catalog_row['DE']
        
        logger.info(f"Generated scene with {len(scene_data['stars'])} stars")
        for i, star in enumerate(scene_data['stars']):
            pos = star['detector_position']
            logger.info(f"  Star {i}: FPA position ({pos[0]:.1f}, {pos[1]:.1f})")
        
        # Step 5: Render scene with PSFs
        logger.info("Step 5: Rendering scene with PSFs...")
        radiometry = multi_star_pipeline.radiometry
        scene_data = radiometry.render_scene(scene_data, psf_data)
        
        detector_shape = scene_data['detector_image'].shape
        total_signal = np.sum(scene_data['detector_image'])
        logger.info(f"Rendered detector image: {detector_shape}, total signal: {total_signal:.0f}")
        
        # Step 6: Detect stars
        logger.info("Step 6: Detecting stars...")
        centroid_results = pipeline.detect_stars_and_calculate_centroids(
            [scene_data['detector_image']], 
            k_sigma=3.0,  # Lower threshold for easier detection
            min_pixels=3,
            max_pixels=200
        )
        
        detected_count = len(centroid_results['centroids'])
        logger.info(f"Detected {detected_count} stars")
        
        if detected_count == 0:
            raise ValueError("No stars detected. Check PSF rendering and detection parameters.")
        
        # Log detected centroids
        for i, centroid in enumerate(centroid_results['centroids']):
            logger.info(f"  Detected centroid {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        # Step 7: Calculate bearing vectors
        logger.info("Step 7: Calculating bearing vectors...")
        bearing_results = pipeline.calculate_bearing_vectors(
            centroid_results['centroids'],
            scene_data['detector_image'].shape,
            sensor_pixel_pitch_um=0.5  # Gen_1 PSF pixel pitch
        )
        
        bearing_vectors = bearing_results['bearing_vectors']
        logger.info(f"Calculated {len(bearing_vectors)} bearing vectors")
        
        # Step 8: Coordinate validation
        logger.info("Step 8: Validating coordinate transformations...")
        validation_results = validate_coordinate_transformations(
            catalog, scene_data, bearing_vectors
        )
        
        log_coordinate_pipeline_details(scene_data, bearing_vectors, validation_results)
        
        # Step 9: BAST matching
        logger.info("Step 9: Executing BAST matching...")
        star_matches = match(
            observed_stars=bearing_vectors,
            catalog=catalog,
            angle_tolerance=0.05,  # 0.05 radian tolerance (~3 degrees)
            min_confidence=0.7
        )
        
        logger.info(f"BAST matching found {len(star_matches)} matches")
        for i, match_result in enumerate(star_matches):
            logger.info(f"  Match {i}: {match_result}")
        
        # Step 10: Overall test validation
        logger.info("Step 10: Validating overall test results...")
        
        test_results = {
            'catalog_stars': len(catalog),
            'detected_stars': detected_count,
            'bearing_vectors': len(bearing_vectors),
            'coordinate_validation': validation_results,
            'bast_matches': len(star_matches),
            'match_details': star_matches
        }
        
        # Determine overall test success
        success_criteria = [
            detected_count >= 3,  # At least 3 stars detected
            len(bearing_vectors) >= 3,  # At least 3 bearing vectors
            validation_results['status'] in ['passed', 'warning'],  # Coordinate validation acceptable
            len(star_matches) >= 1  # At least one BAST match found
        ]
        
        overall_success = all(success_criteria)
        
        logger.info("=" * 50)
        logger.info("END-TO-END TEST RESULTS:")
        logger.info(f"  Catalog stars: {test_results['catalog_stars']}")
        logger.info(f"  Detected stars: {test_results['detected_stars']}")
        logger.info(f"  Bearing vectors: {test_results['bearing_vectors']}")
        logger.info(f"  Coordinate validation: {validation_results['status']}")
        logger.info(f"  BAST matches: {test_results['bast_matches']}")
        logger.info(f"  Overall success: {'PASSED' if overall_success else 'FAILED'}")
        logger.info("=" * 50)
        
        return test_results, overall_success
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def run_coordinate_only_test():
    """
    Run a simplified test focusing only on coordinate transformations.
    """
    logger.info("Starting coordinate-only test...")
    
    try:
        # Create simple 3-star catalog
        catalog_builder = SyntheticCatalogBuilder()
        catalog = catalog_builder.create_triangle_catalog(separation_deg=1.0)
        
        # Create pipeline  
        pipeline = StarTrackerPipeline()
        
        # Test coordinate conversion
        from .scene_generator import MultiStarSceneGenerator
        scene_generator = MultiStarSceneGenerator(pipeline)
        scene_data = scene_generator.generate_scene(catalog)
        
        # Extract RA/Dec positions
        ra_dec_positions = []
        for star in scene_data['stars']:
            catalog_row = catalog.iloc[star['catalog_idx']]
            ra_dec_positions.append((catalog_row['RA'], catalog_row['DE']))
        
        # Calculate true separations using astropy
        from .coordinate_validation import calculate_true_angular_separations
        true_separations = calculate_true_angular_separations(ra_dec_positions)
        
        logger.info("True angular separations from RA/Dec:")
        for i, sep in enumerate(true_separations):
            logger.info(f"  Separation {i}: {sep:.3f} degrees")
        
        return True
        
    except Exception as e:
        logger.error(f"Coordinate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Multi-Star BAST Matching - End-to-End Test")
    logger.info("=" * 50)
    
    # Run coordinate-only test first
    coord_success = run_coordinate_only_test()
    logger.info(f"Coordinate test: {'PASSED' if coord_success else 'FAILED'}")
    
    if coord_success:
        # Run full end-to-end test
        test_results, overall_success = run_triangle_matching_test()
        
        if overall_success:
            logger.info("🎉 END-TO-END TEST PASSED! Multi-star BAST matching is working!")
        else:
            logger.error("❌ END-TO-END TEST FAILED. Check logs for details.")
            sys.exit(1)
    else:
        logger.error("❌ COORDINATE TEST FAILED. Skipping full test.")
        sys.exit(1)