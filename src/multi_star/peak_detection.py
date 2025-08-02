import numpy as np
from scipy.ndimage import maximum_filter, label
import logging

logger = logging.getLogger(__name__)

def detect_stars_peak_method(image, min_intensity_fraction=0.1, min_separation=20, min_pixels=5):
    """
    Detect stars using peak detection instead of adaptive thresholding.
    
    Args:
        image: Input detector image
        min_intensity_fraction: Minimum intensity as fraction of image maximum
        min_separation: Minimum separation between peaks in pixels
        min_pixels: Minimum region size for valid detections
        
    Returns:
        list: List of (x, y) centroid positions
    """
    # Find image statistics
    image_max = np.max(image)
    image_mean = np.mean(image)
    image_std = np.std(image)
    
    logger.info(f"Image stats: max={image_max:.1f}, mean={image_mean:.1f}, std={image_std:.1f}")
    
    # Threshold: either fraction of max or multiple of std above mean
    threshold_max = image_max * min_intensity_fraction
    threshold_std = image_mean + 5 * image_std
    threshold = max(threshold_max, threshold_std)
    
    logger.info(f"Using threshold: {threshold:.1f} (max_frac: {threshold_max:.1f}, std: {threshold_std:.1f})")
    
    # Find local maxima
    local_max = maximum_filter(image, size=min_separation) == image
    
    # Apply threshold
    peaks = local_max & (image > threshold)
    
    # Get peak coordinates
    peak_coords = np.where(peaks)
    peak_intensities = image[peak_coords]
    
    logger.info(f"Found {len(peak_coords[0])} peaks above threshold")
    
    # Sort by intensity (brightest first)
    sorted_indices = np.argsort(peak_intensities)[::-1]
    
    centroids = []
    for i in sorted_indices:
        y, x = peak_coords[0][i], peak_coords[1][i]
        intensity = peak_intensities[i]
        
        # Refine centroid using moment method in local region
        refined_centroid = refine_centroid_moment(image, (x, y), window_size=15)
        
        if refined_centroid is not None:
            centroids.append(refined_centroid)
            logger.info(f"Peak {len(centroids)}: ({refined_centroid[0]:.2f}, {refined_centroid[1]:.2f}) intensity={intensity:.1f}")
    
    return centroids


def refine_centroid_moment(image, initial_pos, window_size=15):
    """
    Refine centroid position using moment method in local window.
    
    Args:
        image: Input image
        initial_pos: Initial (x, y) position
        window_size: Size of local window for moment calculation
        
    Returns:
        tuple: Refined (x, y) centroid position or None if failed
    """
    x0, y0 = initial_pos
    half_window = window_size // 2
    
    # Extract local window
    y_start = max(0, y0 - half_window)
    y_end = min(image.shape[0], y0 + half_window + 1)
    x_start = max(0, x0 - half_window)
    x_end = min(image.shape[1], x0 + half_window + 1)
    
    window = image[y_start:y_end, x_start:x_end]
    
    if window.size == 0 or np.sum(window) == 0:
        return None
    
    # Calculate moments
    y_indices, x_indices = np.mgrid[0:window.shape[0], 0:window.shape[1]]
    
    total_intensity = np.sum(window)
    centroid_x_local = np.sum(x_indices * window) / total_intensity
    centroid_y_local = np.sum(y_indices * window) / total_intensity
    
    # Convert back to global coordinates
    centroid_x = x_start + centroid_x_local
    centroid_y = y_start + centroid_y_local
    
    return (centroid_x, centroid_y)


def create_mock_centroid_results(centroids):
    """
    Create a results dictionary compatible with the existing pipeline.
    
    Args:
        centroids: List of (x, y) centroid positions
        
    Returns:
        dict: Results in the format expected by the pipeline
    """
    return {
        'successful_detections': len(centroids),
        'centroids': centroids,
        'centroid_errors': [0.0] * len(centroids),  # No ground truth available
        'true_center': None,
        'detection_success_rate': 1.0 if centroids else 0.0
    }