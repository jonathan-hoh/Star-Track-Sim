# track.py - bast
#
# Python module for running basic astronomical star tracking (bast)

from typing import Optional, Tuple
import os
from pathlib import Path

import numpy as np
import logging

from calibrate import Calibration
from identify import identify
from match import match, load_star_catalog
from preprocess import process_image as preprocess  # Note the alias
from resolve import resolve

logger = logging.getLogger(__name__)


class Tracker:
    """A class for tracking stellar objects and determining spacecraft orientation."""

    def __init__(
        self,
        cal: Calibration = Calibration(),
        catalog_path: str = None,
        magnitude: float = 4.0,
        fov_degrees: float = 10.0,
    ):
        """Initialize the tracker with optional calibration data and catalog.

        Args:
            cal: Calibration object containing dark frame and distortion map
            catalog_path: Path to star catalog file (.csv or .pickle)
            magnitude: Magnitude threshold for star catalog
            fov_degrees: Field of view in degrees
        """
        self.cal = cal
        self.fov_degrees = fov_degrees

        # Load the catalog
        self._load_star_catalog(catalog_path, magnitude, fov_degrees)

        self.current_quaternion: Optional[np.ndarray] = None

    def _load_star_catalog(
        self,
        catalog_path: str = None,
        magnitude: float = 4.0,
        fov_degrees: float = 10.0,
    ):
        """Load and prepare star catalog for matching.

        If no catalog path is provided, will look for HIPPARCOS.csv in the default location.

        Args:
            catalog_path: Path to catalog file (.csv or .pickle)
            magnitude: Magnitude threshold for filtering stars
            fov_degrees: Field of view in degrees
        """
        # If no path is provided, use the default HIPPARCOS.csv in the package directory
        if catalog_path is None:
            catalog_path = Path(__file__).parent / "catalogs" / "HIPPARCOS.csv"
            if not os.path.exists(catalog_path):
                logger.warning(f"Default catalog not found at {catalog_path}")
                # Create an empty catalog as fallback
                self.catalog = None
                return

        try:
            self.catalog = load_star_catalog(catalog_path, magnitude, fov_degrees)
            logger.info(
                f"Successfully loaded star catalog with {len(self.catalog)} stars and {self.catalog.num_triplets()} triplets"
            )
        except Exception as e:
            logger.error(f"Failed to load catalog: {str(e)}")
            # Create a simplified fallback catalog for testing (as in the original code)
            self.catalog = self._create_fallback_catalog()
            logger.warning("Using fallback catalog with simple orthogonal vectors")

    def _create_fallback_catalog(self):
        """Create a simplified fallback catalog for testing purposes."""
        # This is a very basic catalog with orthogonal vectors, similar to the original implementation
        from catalog import Catalog
        import pandas as pd

        # Create a minimal DataFrame with the required columns
        df = pd.DataFrame(
            {
                "Star ID": [1, 2, 3],
                "RA": [0.0, np.pi / 2, 0.0],
                "DE": [0.0, 0.0, np.pi / 2],
                "Magnitude": [1.0, 1.0, 1.0],
            }
        )

        # Create a catalog with these values
        catalog = Catalog(df, self.fov_degrees)

        # Create basic triplets manually
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        # Calculate angles for the triplet
        from match import calculate_vector_angle

        angle_01 = calculate_vector_angle(vectors[0], vectors[1])
        angle_02 = calculate_vector_angle(vectors[0], vectors[2])
        angle_12 = calculate_vector_angle(vectors[1], vectors[2])

        # Add the triplet to the catalog
        catalog.at[0, "Triplets"] = [(angle_01, angle_02, angle_12)]

        return catalog

    def track(self, image):
        """Process an image to determine spacecraft orientation.

        The tracking process consists of several steps:
        1. Image preprocessing with dark frame correction
        2. Star identification and vector calculation
        3. Matching observed stars with catalog entries
        4. Computing final orientation (rotation matrix and quaternion)

        Args:
            image: Input image to process

        Returns:
            tuple: (rotation_matrix, quaternion) representing spacecraft orientation
        """
        # Step 1: Preprocess the image using dark frame correction if available
        preprocessed = preprocess(image=image, dark_frame=self.cal.dark_frame)

        # Step 2: Convert star positions to bearing vectors, accounting for lens distortion
        bearing_vectors = identify(self.cal.distortion_map, preprocessed)

        # Step 3: Match observed stars with catalog entries
        # Uses previous quaternion if available for better matching
        star_matches = match(
            observed_stars=bearing_vectors,
            catalog=self.catalog,
            prior_quaternion=self.current_quaternion,
        )

        # Step 4: Compute final orientation from matched stars
        if not star_matches:
            logger.warning("No star matches found, unable to determine attitude")
            return None, None

        # Extract catalog vectors for matched stars
        catalog_vectors = []
        for match_obj in star_matches:
            # Get the star ID from the catalog
            star_id = self.catalog.iloc[match_obj.catalog_idx]["Star ID"]

            # Get the corresponding vector from the catalog
            # For real catalog data, this would be derived from RA/Dec
            # Here we're getting the unit vector based on RA/Dec coordinates
            ra = self.catalog.iloc[match_obj.catalog_idx]["RA"]
            dec = self.catalog.iloc[match_obj.catalog_idx]["DE"]

            # Convert RA/Dec to unit vector
            x = np.cos(dec) * np.cos(ra)
            y = np.cos(dec) * np.sin(ra)
            z = np.sin(dec)

            catalog_vectors.append(np.array([x, y, z]))

        # If we don't have enough matches for attitude determination, return None
        if len(catalog_vectors) < 2:
            logger.warning("Insufficient matches for attitude determination")
            return None, None

        # Convert matched star IDs to indices list for resolve
        matched_indices = [
            (m.catalog_idx, m.observed_idx, m.reference_idx) for m in star_matches
        ]

        try:
            # Call resolve with our matched stars
            quaternion, rotation_matrix = resolve(
                observed_vectors=bearing_vectors,
                matched_indices=matched_indices,
                catalog_vectors=np.array(catalog_vectors),
            )

            # Store current quaternion for next iteration
            self.current_quaternion = quaternion

            return rotation_matrix, quaternion

        except Exception as e:
            logger.error(f"Failed to resolve attitude: {str(e)}")
            return None, None
