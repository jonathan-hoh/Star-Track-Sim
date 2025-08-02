import numpy as np
import cv2
from typing import List, Tuple
import logging

# Configure logger at module level
logger = logging.getLogger(__name__)


def group_pixels(
    thresholded_image: np.ndarray, min_pixels: int = 3, max_pixels: int = 100
) -> List[np.ndarray]:
    """
    Group connected pixels into star regions

    Args:
        thresholded_image: Binary image from preprocess.py output
        min_pixels: Minimum pixel count for valid star (default 3)
        max_pixels: Maximum pixel count for valid star (default 100)

    Returns:
        List of binary masks, each representing a star region.
        Regions must have pixel count in range [min_pixels, max_pixels].
    """
    # Ensure input is binary
    binary_image = (thresholded_image > 0).astype(np.uint8)

    # Find connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    # Filter regions by size (label 0 is background)
    valid_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Strict size filtering - changed from >= to > for min_pixels
        if area > min_pixels and area <= max_pixels:
            # Create binary mask for this region
            region_mask = labels == i
            valid_regions.append(region_mask)

    return valid_regions


def calculate_centroid(
    star_region: np.ndarray, original_image: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate centroid using modified moment algorithm

    Args:
        star_region: Binary mask of star region
        original_image: Original pixel values for intensity weighting

    Returns:
        Tuple of (x_centroid, y_centroid, total_intensity)
    """
    # Get pixel values in region
    region_intensities = original_image * star_region

    # Calculate total intensity
    total_intensity = np.sum(region_intensities)

    if total_intensity <= 0:
        return None

    # Get coordinate arrays
    y_coords, x_coords = np.nonzero(star_region)

    # Calculate intensity-weighted moments
    x_moment = np.sum(x_coords * region_intensities[y_coords, x_coords])
    y_moment = np.sum(y_coords * region_intensities[y_coords, x_coords])

    # Calculate centroid coordinates
    x_centroid = x_moment / total_intensity
    y_centroid = y_moment / total_intensity

    return x_centroid, y_centroid, total_intensity


def apply_distortion_correction(
    centroids: List[Tuple[float, float, float]], distortion_map: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Apply distortion correction to centroid positions

    Args:
        centroids: List of (x, y, intensity) centroid positions
        distortion_map: Distortion polynomial coefficients

    Returns:
        List of corrected (x, y) positions
    """
    # For now, implementing simplest radial distortion model
    # k1 is first radial distortion coefficient
    k1 = distortion_map[0] if distortion_map is not None else 0

    corrected_positions = []
    for x, y, _ in centroids:
        # Calculate radius from center
        r = np.sqrt(x * x + y * y)

        # Apply radial distortion correction
        correction_factor = 1 + k1 * r * r

        # Correct coordinates
        x_corrected = x * correction_factor
        y_corrected = y * correction_factor

        corrected_positions.append((x_corrected, y_corrected))

    return corrected_positions


def calculate_bearing_vectors(
    corrected_positions: List[Tuple[float, float]], focal_length: float
) -> List[np.ndarray]:
    """
    Transform image coordinates to bearing vectors

    Args:
        corrected_positions: List of distortion-corrected (x, y) positions
        focal_length: Camera focal length in pixels

    Returns:
        List of unit vectors pointing to stars
    """
    bearing_vectors = []

    for x, y in corrected_positions:
        # Convert to focal plane coordinates
        x_focal = x / focal_length
        y_focal = y / focal_length

        # Create bearing vector [1, x_focal, y_focal]
        vector = np.array([1.0, x_focal, y_focal])

        # Normalize to unit vector
        vector = vector / np.linalg.norm(vector)

        bearing_vectors.append(vector)

    return bearing_vectors


def identify(
    distortion_map: np.ndarray,
    thresholded_image: np.ndarray,
    focal_length: float = 2048.0,
) -> List[np.ndarray]:
    """
    Main identification function - process thresholded image into bearing vectors

    Args:
        distortion_map: Camera distortion coefficients
        thresholded_image: Output from preprocess.py
        focal_length: Camera focal length in pixels

    Returns:
        List of unit vectors pointing to identified stars
    """
    # Group pixels into star regions
    star_regions = group_pixels(thresholded_image)

    # Calculate centroids
    centroids = []
    for region in star_regions:
        centroid = calculate_centroid(region, thresholded_image)
        if centroid is not None:
            centroids.append(centroid)

    logger.info(f"Calculated {len(centroids)} centroids")

    # Apply distortion correction
    corrected_positions = apply_distortion_correction(centroids, distortion_map)

    # Transform to bearing vectors
    bearing_vectors = calculate_bearing_vectors(corrected_positions, focal_length)

    logger.info("Star identification complete")
    return bearing_vectors


if __name__ == "__main__":

    # Configure root logger when running as script
    logging.basicConfig(level=logging.INFO)

    import sys
    from astropy.io import fits

    if len(sys.argv) < 2:
        print("Usage: python identify.py <thresholded_image> [distortion_map]")
        sys.exit(1)

    # Load thresholded image
    image_path = sys.argv[1]
    try:
        with fits.open(image_path) as hdul:
            thresholded_image = hdul[0].data
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        sys.exit(1)

    # Load distortion map if provided
    distortion_map = None
    if len(sys.argv) > 2:
        try:
            distortion_map = np.load(sys.argv[2])
        except Exception as e:
            logger.error(f"Error loading distortion map: {str(e)}")
            sys.exit(1)

    # Process image
    bearing_vectors = identify(distortion_map, thresholded_image)

    # Save output
    output_path = "bearing_vectors.npy"
    np.save(output_path, np.array(bearing_vectors))
    logger.info(f"Saved {len(bearing_vectors)} bearing vectors to {output_path}")
