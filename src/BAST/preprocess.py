import numpy as np
import cv2
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def verify_image(image: np.ndarray):
    """Verify that image has proper dimensions.

    Args:
        image (np.ndarray): Input image to verify

    Raises:
        ValueError: If image dimensions or dtype are invalid
    """
    if image.shape != (2048, 2048):
        raise ValueError(f"Image must be 2048x2048, got {image.shape}")

    if image.dtype != np.uint16:
        raise ValueError(f"Image must be 12-bit (uint16), got {image.dtype}")


def apply_dark_correction(
    image: np.ndarray, dark_frame: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply dark frame correction to image. Can handle image with or without dark frame calibrator.
    Can also handle case where series of dark frames are used for calibration.

    Args:
        image (np.ndarray): Input image to correct
        dark_frame (Optional[np.ndarray], optional): Dark frame for correction. Defaults to None.

    Raises:
        ValueError: If dark frame dimensions don't match image

    Returns:
        np.ndarray: Dark-corrected image
    """
    if dark_frame is None:
        logger.warning("No dark frame provided, skipping dark correction")
        return image.astype(np.float32)

    if dark_frame.shape != image.shape:
        raise ValueError(f"Dark frame must be 2048x2048, got {dark_frame.shape}")

    corrected = image.astype(np.float32) - dark_frame.astype(np.float32)
    return np.clip(corrected, 0, None)


def calculate_background_statistics(
    image: np.ndarray, block_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation for each 32x32 pixel group in the image.

    Args:
        image (np.ndarray): Input image
        block_size (int, optional): Size of blocks to analyze. Defaults to 32.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays containing local means and standard deviations
    """
    rows, cols = image.shape

    pad_rows = block_size - (rows % block_size) if rows % block_size != 0 else 0
    pad_cols = block_size - (cols % block_size) if cols % block_size != 0 else 0

    if pad_rows > 0 or pad_cols > 0:
        image = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode="reflect")

    num_blocks_y = image.shape[0] // block_size
    num_blocks_x = image.shape[1] // block_size

    local_mean = np.zeros((num_blocks_y, num_blocks_x))
    local_std = np.zeros((num_blocks_y, num_blocks_x))

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = image[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            local_mean[i, j] = np.mean(block)
            local_std[i, j] = np.std(block)

    return local_mean, local_std


def apply_threshold(
    image: np.ndarray,
    local_mean: np.ndarray,
    local_std: np.ndarray,
    k_threshold: float = 4.0,
) -> np.ndarray:
    """Find pixels that are 4σ (or more) brighter than μ_local. Number of σ can be adjusted but 4 is theoretically optimal.

    Args:
        image (np.ndarray): Input image
        local_mean (np.ndarray): Array of local means
        local_std (np.ndarray): Array of local standard deviations
        k_threshold (float, optional): Sigma multiplier for threshold. Defaults to 4.0.

    Returns:
        np.ndarray: Thresholded image
    """
    local_mean = local_mean.astype(np.float32)
    local_std = local_std.astype(np.float32)

    mean_upsampled = cv2.resize(
        local_mean, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    std_upsampled = cv2.resize(
        local_std, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    threshold = mean_upsampled + k_threshold * std_upsampled
    thresholded = np.where(image > threshold, image, 0)

    return thresholded


def create_dark_frame(calibration_images: list) -> np.ndarray:
    """Take median of multiple dark frames to create calibration frame.

    Args:
        calibration_images (list): List of dark frame images

    Raises:
        ValueError: If no calibration images provided

    Returns:
        np.ndarray: Median dark frame
    """
    if not calibration_images:
        raise ValueError("No calibration images provided")

    stacked = np.stack(calibration_images)
    dark_frame = np.median(stacked, axis=0)

    return dark_frame.astype(np.float32)


def process_image(
    image: np.ndarray,
    dark_frame: Optional[np.ndarray] = None,
    block_size: int = 32,
    k_threshold: float = 4.0,
) -> np.ndarray:
    """Main pipeline for processing astronomical images.

    Args:
        image (np.ndarray): Input image to process
        dark_frame (Optional[np.ndarray], optional): Dark frame for correction. Defaults to None.
        block_size (int, optional): Size of analysis blocks. Defaults to 32.
        k_threshold (float, optional): Sigma threshold. Defaults to 4.0.

    Returns:
        np.ndarray: Processed image
    """
    verify_image(image)
    corrected_image = apply_dark_correction(image, dark_frame)
    local_mean, local_std = calculate_background_statistics(corrected_image, block_size)
    thresholded_image = apply_threshold(
        corrected_image, local_mean, local_std, k_threshold
    )

    logger.info("Image preprocessing complete")
    return thresholded_image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path> [dark_frame_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    dark_frame = None
    if len(sys.argv) > 2:
        dark_frame_path = sys.argv[2]
        dark_frame = cv2.imread(dark_frame_path, cv2.IMREAD_ANYDEPTH)

    try:
        thresholded_image = process_image(image, dark_frame)
        output_path = "thresholded_output.fits"
        from astropy.io import fits

        fits.writeto(output_path, thresholded_image, overwrite=True)
        logger.info(f"Processed image saved to {output_path}")

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        sys.exit(1)
