# resolve.py
#!/usr/bin/env python3
"""
resolve.py - Attitude determination module using QUEST algorithm
This module takes bearing vectors and matched catalog stars to determine spacecraft attitude.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import logging
import os
import sys
from datetime import datetime
from .match import StarMatch

# Define a small value to avoid division by zero
VIRTUAL_ZERO = 1e-10

logger = logging.getLogger(__name__)


def build_davenport_matrix(
    observed_vectors: List[np.ndarray],
    catalog_vectors: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Build Davenport K-matrix for QUEST algorithm.

    Args:
        observed_vectors: List of observed unit vectors
        catalog_vectors: List of corresponding catalog unit vectors
        weights: Optional weights for each measurement (default: equal weights)

    Returns:
        4x4 Davenport K-matrix
    """
    if weights is None:
        weights = [1.0] * len(observed_vectors)

    # Initialize the attitude profile matrix B
    B = np.zeros((3, 3))

    # Build attitude profile matrix
    for b_i, r_i, w_i in zip(observed_vectors, catalog_vectors, weights):
        # Convert vectors to column vectors for outer product
        b_col = b_i.reshape(3, 1)
        r_col = r_i.reshape(3, 1)

        # Add weighted outer product to B
        B += w_i * (r_col @ b_col.T)

    # Calculate key components using custom trace
    sigma = np.linalg.trace(B)
    S = B + B.T

    # Calculate Z vector (adjoint matrix elements)
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])

    # Build the K matrix
    K = np.zeros((4, 4))

    # Fill in K matrix components
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z

    # Fill in lower 3x3 block
    K[1:, 1:] = S - sigma * np.eye(3)

    return K  # Return only the K matrix


def quest_algorithm(K: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve for optimal quaternion using QUEST algorithm.
    Returns quaternion in scalar-first format [w, x, y, z].

    Args:
        K: 4x4 Davenport matrix from build_davenport_matrix

    Returns:
        Tuple of (optimal quaternion, maximum eigenvalue)

    Raises:
        ValueError: If K is not a 4x4 matrix
        RuntimeError: If eigenvalue computation fails
    """
    # Input validation
    if K.shape != (4, 4):
        raise ValueError(f"K matrix must be 4x4, got shape {K.shape}")

    try:
        # Compute eigenvalues and eigenvectors of K
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # Find index of maximum eigenvalue
        max_index = np.argmax(eigenvalues.real)

        # Get the corresponding eigenvector
        max_eigenvector = eigenvectors[:, max_index]

        # Ensure the quaternion has real components
        quaternion = max_eigenvector.real

        # The eigenvalue represents the quality of the solution
        max_eigenvalue = eigenvalues[max_index].real

        # Normalize the quaternion
        norm = np.linalg.norm(quaternion)
        if norm < VIRTUAL_ZERO:  # Check for near-zero quaternion
            raise RuntimeError("Failed to compute valid quaternion (zero norm)")
        quaternion = quaternion / norm

        # IMPORTANT: The quaternion should have positive scalar component (w)
        # If w is negative, negate the quaternion (this represents the same rotation)
        if quaternion[0] < 0:
            logger.warning("Negating quaternion to ensure positive scalar component")
            quaternion = -quaternion
        return quaternion, max_eigenvalue

    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Failed to compute eigenvalues: {str(e)}")


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix (Direction Cosine Matrix).
    Takes quaternion in scalar-first format [w, x, y, z].

    Args:
        q: Quaternion in [w, x, y, z] format where w is scalar component

    Returns:
        3x3 rotation matrix

    Raises:
        ValueError: If quaternion is not normalized or wrong size
    """
    # Input validation
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have 4 elements, got shape {q.shape}")

    # Check if quaternion is normalized
    if not np.isclose(np.linalg.norm(q), 1.0, rtol=1e-5):
        # 1e-5 is strict enough to catch actual normalization issues while being
        # lenient enough to handle typical floating-point rounding errors
        raise ValueError("Quaternion must be normalized")

    # Extract components
    w, x, y, z = q

    # Method 1: Direct analytical formula
    # This is more computationally efficient
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )

    return R


def resolve(
    observed_vectors: List[np.ndarray],
    matched_indices: List[Union[Tuple[int, int, int], StarMatch]],
    catalog_vectors: np.ndarray,
    min_eigenvalue: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main attitude determination function.

    Args:
        observed_vectors: List of observed unit vectors
        matched_indices: List of matched star indices or StarMatch objects
        catalog_vectors: Array of catalog star vectors
        min_eigenvalue: Minimum eigenvalue threshold for solution confidence (default: 0.5)

    Returns:
        Tuple of (quaternion, rotation_matrix)

    Raises:
        ValueError: If insufficient matches provided
        RuntimeError: If attitude determination fails or solution confidence is low
    """
    # Input validation
    if len(matched_indices) < 2:
        raise ValueError("Insufficient star matches for attitude determination")

    # Extract matched catalog vectors
    matched_catalog_vectors = []
    matched_observed_vectors = []

    # Process each matched star
    for match in matched_indices:
        # Handle both tuple format and StarMatch object format
        if isinstance(match, StarMatch):
            cat_idx = match.catalog_idx
            obs_idx = match.observed_idx
        else:
            cat_idx = match[0]
            obs_idx = match[1]

        # Depending on the catalog format, extract the catalog vector
        if isinstance(catalog_vectors, list):
            cat_vector = catalog_vectors[cat_idx]
        else:
            # Handle numpy array format
            cat_vector = catalog_vectors[cat_idx]

        obs_vector = observed_vectors[obs_idx]

        # Ensure vectors are normalized
        cat_norm = np.linalg.norm(cat_vector)
        obs_norm = np.linalg.norm(obs_vector)

        if cat_norm < VIRTUAL_ZERO or obs_norm < VIRTUAL_ZERO:
            raise RuntimeError("Zero or near-zero length vector encountered")

        cat_vector = cat_vector / cat_norm
        obs_vector = obs_vector / obs_norm

        matched_catalog_vectors.append(cat_vector)
        matched_observed_vectors.append(obs_vector)

    # Check for nearly parallel vectors
    if len(matched_catalog_vectors) >= 2:
        for i in range(len(matched_catalog_vectors)):
            for j in range(i + 1, len(matched_catalog_vectors)):
                # Calculate angle between vector pairs
                cat_angle = np.arccos(
                    np.clip(
                        np.dot(matched_catalog_vectors[i], matched_catalog_vectors[j]),
                        -1.0,
                        1.0,
                    )
                )
                obs_angle = np.arccos(
                    np.clip(
                        np.dot(
                            matched_observed_vectors[i], matched_observed_vectors[j]
                        ),
                        -1.0,
                        1.0,
                    )
                )

                # If angles are too small, vectors are nearly parallel
                if abs(cat_angle) < 1e-6 or abs(obs_angle) < 1e-6:
                    # Micro-radian considered the general threshold confirmation
                    raise RuntimeError("Nearly parallel vectors detected")
    else:
        raise RuntimeError("Insufficient catalog vectors for attitude determination")

    try:
        # Build Davenport matrix
        K = build_davenport_matrix(matched_observed_vectors, matched_catalog_vectors)

        # Get quaternion from QUEST algorithm
        quaternion, max_eigenvalue = quest_algorithm(K)

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_matrix(quaternion)

        # Verify solution quality
        if max_eigenvalue < min_eigenvalue:
            raise RuntimeError(
                f"Low confidence in attitude solution (eigenvalue: {max_eigenvalue:.3f} < {min_eigenvalue:.3f})"
            )

        return quaternion, rotation_matrix

    except (ValueError, RuntimeError) as e:
        # Only catch the specific exceptions we expect from our functions:
        # ValueError: from input validation in quaternion_to_matrix
        # RuntimeError: from quest_algorithm and our explicit raises
        logger.error(f"Failed to determine attitude: {str(e)}")
        raise RuntimeError(f"Attitude determination failed: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_filename = f"logs/star_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Determine spacecraft attitude from star matches"
    )
    parser.add_argument(
        "bearing_vectors", help="Path to numpy file containing bearing vectors"
    )
    parser.add_argument(
        "matched_indices", help="Path to numpy file containing matched indices"
    )
    parser.add_argument("catalog", help="Path to star catalog file")
    parser.add_argument(
        "--output",
        "-o",
        default="attitude.npz",
        help="Output file for attitude quaternion and rotation matrix",
    )

    args = parser.parse_args()

    try:
        # Load inputs
        observed_vectors = np.load(args.bearing_vectors)
        matched_indices = np.load(args.matched_indices)
        catalog_vectors = np.load(args.catalog)

        logger.info(
            f"Loaded {len(observed_vectors)} bearing vectors and "
            f"{len(matched_indices)} matched stars"
        )

        # Compute attitude
        quaternion, rotation_matrix = resolve(
            observed_vectors, matched_indices, catalog_vectors
        )

        # Save results
        np.savez(args.output, quaternion=quaternion, rotation_matrix=rotation_matrix)

        # Print results
        logger.info(f"Attitude quaternion [w,x,y,z]: {quaternion}")
        logger.info(f"Rotation matrix:\n{rotation_matrix}")
        logger.info(f"Results saved to {args.output}")

        # Optional: Print Euler angles for visualization
        euler_angles = np.array(
            [
                np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]),  # yaw
                np.arcsin(-rotation_matrix[2, 0]),  # pitch
                np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),  # roll
            ]
        )
        euler_degrees = np.degrees(euler_angles)
        logger.info(f"Euler angles [yaw,pitch,roll] (degrees): {euler_degrees}")

    except Exception as e:
        logger.error(f"Error in attitude determination: {str(e)}")
        sys.exit(1)

    # Add some basic testing
    if len(sys.argv) == 1:  # No arguments provided, run test
        logger.info("Running test case...")

        # Create simple test case
        test_observed = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        test_catalog = np.array(test_observed)  # Same vectors for simple test
        test_matches = [
            (0, 0, 2),
            (1, 1, 2),
        ]  # Two matches using third star as reference

        q, R = resolve(test_observed, test_matches, test_catalog)
        logger.info("Test quaternion:")
        logger.info(q)
        logger.info("Test rotation matrix:")
        logger.info(R)
