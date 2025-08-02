#!/usr/bin/env python3
"""
match.py – Pyramid-based star matching module for low-cost star trackers.
This module implements a pyramid algorithm to match observed stars to catalog stars.
It finds multiple star matches (up to 4) to improve attitude determination accuracy.
"""

import logging
import os
import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import tqdm

from .catalog import from_csv, from_pickle  # Import the catalog module

logger = logging.getLogger(__name__)

MIN_STARS_FOR_TRIPLET = 3
MAX_MATCHES = 4  # Maximum number of matches to return


class StarMatch:
    """Represents the result of a star matching operation."""

    def __init__(
        self,
        catalog_idx: int,
        observed_idx: int,
        reference_idx: int,
        confidence: float = 1.0,
    ):
        """
        Initialize a star match.

        Args:
            catalog_idx: Index of matched star in catalog
            observed_idx: Index of matched star in observed set
            reference_idx: Index of reference star used for matching
            confidence: Match confidence score (0-1)
        """
        self.catalog_idx = catalog_idx
        self.observed_idx = observed_idx
        self.reference_idx = reference_idx
        self.confidence = confidence
        self.is_matched = True

    def __bool__(self) -> bool:
        """Allows for simple truthiness checking of match results."""
        return self.is_matched

    def __str__(self) -> str:
        if self.is_matched:
            return f"StarMatch(cat={self.catalog_idx}, obs={self.observed_idx}, ref={self.reference_idx}, conf={self.confidence:.3f})"
        return "StarMatch(no match)"

    def __repr__(self) -> str:
        return self.__str__()


def calculate_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle (in radians) between two unit vectors using the dot product."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion into a 3x3 Direction Cosine Matrix (DCM)."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


def score_match(angle_diffs: List[float], base_tolerance: float = 0.01) -> float:
    """
    Calculate confidence score for a match based on angle differences.

    Args:
        angle_diffs: List of angular differences between observed and catalog angles
        base_tolerance: Base angular tolerance in radians

    Returns:
        float: Confidence score between 0 and 1
    """
    scores = [max(0, 1 - abs(diff) / base_tolerance) for diff in angle_diffs]
    return np.mean(scores)


def find_triplet_match(
    observed_triplet: List[np.ndarray],
    catalog: Any,  # Changed to accept a catalog object instead of raw triplets
    i: int,
    j: int,
    k: int,
    angle_tolerance: float = 0.01,
    min_confidence: float = 0.8,
) -> Optional[StarMatch]:
    """Find a match for a single observed triplet in catalog.

    Args:
        observed_triplet: List of 3 observed star vectors
        catalog: Catalog object containing star triplets
        i, j, k: Indices of observed stars
        angle_tolerance: Maximum allowed angle difference
        min_confidence: Minimum confidence threshold

    Returns:
        Optional[StarMatch]: Match result or None if no match found
    """
    # Calculate angles for observed triplet
    angle_01 = calculate_vector_angle(observed_triplet[0], observed_triplet[1])
    angle_02 = calculate_vector_angle(observed_triplet[0], observed_triplet[2])
    angle_12 = calculate_vector_angle(observed_triplet[1], observed_triplet[2])

    # Get the triplets from the catalog
    # Note: catalog["Triplets"] would access the Triplets column from the DataFrame
    catalog_triplets = []

    # Iterate through catalog entries with triplets
    for idx, row in catalog.iterrows():
        if row["Triplets"] is None:
            continue

        # Each entry in Triplets is a list of tuples, where each tuple represents a triplet
        # with 3 angles: (angle_01, angle_02, angle_12)
        for triplet in row["Triplets"]:
            cat_angle_01, cat_angle_02, cat_angle_12 = triplet

            # Calculate differences
            diffs = [
                abs(angle_01 - cat_angle_01),
                abs(angle_02 - cat_angle_02),
                abs(angle_12 - cat_angle_12),
            ]

            # Check if all angles match within tolerance
            if all(diff < angle_tolerance for diff in diffs):
                confidence = score_match(diffs, angle_tolerance)
                if confidence >= min_confidence:
                    return StarMatch(idx, i, k, confidence)

    return None


def match(
    observed_stars: List[np.ndarray],
    catalog: Any,  # Changed to accept a catalog object
    prior_quaternion: Optional[np.ndarray] = None,
    max_matches: int = MAX_MATCHES,
    angle_tolerance: float = 0.01,
    min_confidence: float = 0.8,
) -> List[StarMatch]:
    """
    Main star matching interface. Finds multiple star matches up to max_matches.

    Args:
        observed_stars: List of observed unit vectors
        catalog: Catalog object containing star triplets
        prior_quaternion: Optional prior attitude estimate
        max_matches: Maximum number of matches to return
        angle_tolerance: Maximum allowed angle difference
        min_confidence: Minimum confidence threshold

    Returns:
        List[StarMatch]: List of matches sorted by confidence
    """
    n = len(observed_stars)
    if n < MIN_STARS_FOR_TRIPLET:
        logger.warning("Not enough observed stars to form a triplet")
        return []

    matches = []
    used_observed_indices = set()
    used_catalog_indices = set()

    # Try all possible triplet combinations
    for i in range(n - 2):
        if len(matches) >= max_matches:
            break

        for j in range(i + 1, n - 1):
            if len(matches) >= max_matches:
                break

            for k in range(j + 1, n):
                # Skip if we've already used these stars
                if (
                    i in used_observed_indices
                    or j in used_observed_indices
                    or k in used_observed_indices
                ):
                    continue

                obs_triplet = [observed_stars[i], observed_stars[j], observed_stars[k]]
                match_result = find_triplet_match(
                    obs_triplet,
                    catalog,
                    i,
                    j,
                    k,
                    angle_tolerance,
                    min_confidence,
                )

                if match_result:
                    # Add to used indices to avoid reusing stars
                    used_observed_indices.update([i, j, k])
                    used_catalog_indices.add(match_result.catalog_idx)
                    matches.append(match_result)

                    if len(matches) >= max_matches:
                        break

    # Sort matches by confidence
    matches.sort(key=lambda x: x.confidence, reverse=True)

    if matches:
        logger.info(f"Found {len(matches)} star matches")
        for m in matches:
            logger.debug(f"Match: {m}")
    else:
        logger.warning("No star matches found")

    return matches


def load_star_catalog(
    catalog_path: str, magnitude_threshold: float = 4.0, fov_degrees: float = 10.0
):
    """
    Load and prepare star catalog for matching.

    Args:
        catalog_path: Path to catalog file (.csv or .pickle)
        magnitude_threshold: Maximum star magnitude to include
        fov_degrees: Field of view in degrees

    Returns:
        Catalog object ready for matching
    """
    # Check if it's a pickle file (pre-processed catalog)
    if catalog_path.endswith(".pickle"):
        logger.info(f"Loading pre-processed catalog from {catalog_path}")
        catalog = from_pickle(catalog_path)
    else:
        # Assume it's a CSV file and process it
        logger.info(
            f"Processing catalog from {catalog_path} with magnitude threshold {magnitude_threshold}"
        )
        catalog = from_csv(catalog_path, magnitude_threshold, fov_degrees)

        # Save processed catalog for future use
        output_path = f"{os.path.splitext(catalog_path)[0]}_{magnitude_threshold}_{fov_degrees}.pickle"
        catalog.save(output_path)
        logger.info(f"Saved processed catalog to {output_path}")

    logger.info(f"Loaded catalog with {len(catalog)} stars")
    logger.info(f"Catalog contains {catalog.num_triplets()} triplets")

    return catalog


if __name__ == "__main__":
    # Setup logging
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_filename = f"logs/star_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )

    # Use the catalog module to load and prepare the star catalog
    import argparse

    parser = argparse.ArgumentParser(description="Test star matching with catalog")
    parser.add_argument(
        "--catalog", default="HIPPARCOS.csv", help="Path to catalog file"
    )
    parser.add_argument(
        "--magnitude", type=float, default=4.0, help="Magnitude threshold"
    )
    parser.add_argument(
        "--fov", type=float, default=10.0, help="Field of view in degrees"
    )
    args = parser.parse_args()

    # Load the catalog
    star_catalog = load_star_catalog(args.catalog, args.magnitude, args.fov)

    # Test case
    observed_stars = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.7071, 0.7071, 0.0]),
    ]

    matches = match(observed_stars, star_catalog)
    if matches:
        logger.info(f"Found {len(matches)} matches:")
        for m in matches:
            logger.info(str(m))
    else:
        logger.info("No matches found")
