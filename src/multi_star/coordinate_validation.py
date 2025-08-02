import numpy as np
import logging
from astropy.coordinates import SkyCoord
from astropy import units as u

logger = logging.getLogger(__name__)

def validate_coordinate_transformations(synthetic_catalog, scene_data, bearing_vectors):
    """
    Validate that coordinate transformations preserve angular relationships.
    
    This function ensures that:
    1. RA/Dec → catalog inner-angles (via BAST)  
    2. RA/Dec → FPA positions → bearing vectors → observed inner-angles
    Produce consistent results.
    
    Args:
        synthetic_catalog: BAST Catalog with computed triplets
        scene_data: Scene data with star positions  
        bearing_vectors: List of observed bearing vectors
        
    Returns:
        dict: Validation results with errors and statistics
    """
    validation_results = {
        'status': 'unknown',
        'catalog_angles': [],
        'observed_angles': [],
        'angle_errors': [],
        'max_error_deg': 0.0,
        'mean_error_deg': 0.0,
        'details': []
    }
    
    try:
        # Extract catalog angular separations from first triplet
        if len(synthetic_catalog) == 0 or synthetic_catalog.iloc[0]['Triplets'] is None:
            validation_results['status'] = 'failed'
            validation_results['details'].append('No catalog triplets found')
            return validation_results
            
        catalog_triplet = synthetic_catalog.iloc[0]['Triplets'][0]  # First triplet
        catalog_angles = list(catalog_triplet)  # (angle_01, angle_02, angle_12)
        
        # Calculate observed angles from bearing vectors
        if len(bearing_vectors) < 3:
            validation_results['status'] = 'failed'
            validation_results['details'].append(f'Insufficient bearing vectors: {len(bearing_vectors)}')
            return validation_results
            
        observed_angles = []
        for i in range(3):
            for j in range(i+1, 3):
                # Calculate angle between bearing vectors (in radians)
                dot_product = np.clip(np.dot(bearing_vectors[i], bearing_vectors[j]), -1.0, 1.0)
                angle_rad = np.arccos(dot_product)
                angle_deg = np.degrees(angle_rad)
                observed_angles.append(angle_deg)
        
        # Compare catalog vs observed angles
        angle_errors = []
        for i, (cat_angle, obs_angle) in enumerate(zip(catalog_angles, observed_angles)):
            error = abs(cat_angle - obs_angle)
            angle_errors.append(error)
            logger.info(f"Angle {i}: catalog={cat_angle:.3f}°, observed={obs_angle:.3f}°, error={error:.3f}°")
        
        # Calculate statistics
        max_error = max(angle_errors)
        mean_error = np.mean(angle_errors)
        
        # Determine validation status
        if max_error < 0.1:  # Within 0.1 degree tolerance
            status = 'passed'
        elif max_error < 1.0:  # Within 1 degree tolerance
            status = 'warning'
        else:
            status = 'failed'
            
        validation_results.update({
            'status': status,
            'catalog_angles': catalog_angles,
            'observed_angles': observed_angles,
            'angle_errors': angle_errors,
            'max_error_deg': max_error,
            'mean_error_deg': mean_error,
            'details': [f'Max error: {max_error:.3f}°, Mean error: {mean_error:.3f}°']
        })
        
        logger.info(f"Coordinate validation {status}: max_error={max_error:.3f}°, mean_error={mean_error:.3f}°")
        
    except Exception as e:
        validation_results['status'] = 'error'
        validation_results['details'].append(f'Validation error: {str(e)}')
        logger.error(f"Coordinate validation error: {e}")
    
    return validation_results


def calculate_true_angular_separations(ra_dec_positions):
    """
    Calculate true angular separations from RA/Dec positions using astropy.
    
    Args:
        ra_dec_positions: List of (RA, DE) tuples in radians
        
    Returns:
        list: Angular separations in degrees for all star pairs
    """
    if len(ra_dec_positions) < 2:
        return []
        
    # Create SkyCoord objects
    coords = []
    for ra, dec in ra_dec_positions:
        coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
        coords.append(coord)
    
    # Calculate all pairwise separations
    separations = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            sep = coords[i].separation(coords[j])
            separations.append(sep.deg)
    
    return separations


def log_coordinate_pipeline_details(scene_data, bearing_vectors, validation_results):
    """
    Log detailed information about the coordinate transformation pipeline.
    
    Args:
        scene_data: Scene data with star positions
        bearing_vectors: Observed bearing vectors
        validation_results: Results from coordinate validation
    """
    logger.info("=== Coordinate Pipeline Details ===")
    
    # Log star positions
    for i, star in enumerate(scene_data['stars']):
        ra_deg = np.degrees(star.get('ra', 0))
        dec_deg = np.degrees(star.get('dec', 0))
        det_pos = star['detector_position']
        
        logger.info(f"Star {i}: RA={ra_deg:.3f}°, Dec={dec_deg:.3f}°, "
                   f"FPA=({det_pos[0]:.1f}, {det_pos[1]:.1f})")
    
    # Log bearing vectors  
    for i, bv in enumerate(bearing_vectors):
        logger.info(f"Bearing Vector {i}: [{bv[0]:.6f}, {bv[1]:.6f}, {bv[2]:.6f}]")
    
    # Log validation summary
    if 'catalog_angles' in validation_results and 'observed_angles' in validation_results:
        logger.info("Angle Comparison:")
        for i, (cat, obs, err) in enumerate(zip(
            validation_results['catalog_angles'],
            validation_results['observed_angles'], 
            validation_results['angle_errors']
        )):
            logger.info(f"  Angle {i}: catalog={cat:.3f}°, observed={obs:.3f}°, error={err:.3f}°")
    
    logger.info(f"Validation Status: {validation_results['status']}")
    logger.info("=" * 35)