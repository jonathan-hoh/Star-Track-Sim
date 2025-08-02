#!/usr/bin/env python3
# star_tracker_pipeline.py - Streamlined star tracker centroiding pipeline

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import logging
from pathlib import Path
import time
import glob
from scipy import ndimage
import cv2 # Added for adaptive thresholding
from skimage.measure import block_reduce # Added for robust PSF projection

# Import core modules
from .starcamera_model import star, scene, star_camera, calculate_optical_signal
from .psf_plot import parse_psf_file
from .psf_photon_simulation import simulate_psf_with_poisson_noise
from ..BAST.identify import calculate_centroid, group_pixels # group_pixels uses cv2.connectedComponentsWithStats

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global dictionary for PSF centroid offsets (in millimeters)
# User needs to fill this with actual values.
# Example: centroid_offsets = { "0": (x_offset_mm, y_offset_mm), "1": ( ... ), ... }
# The keys are string representations of the starting numbers of PSF filenames.
centroid_offsets = {
    "0": (-4.25325608E-22,   7.92844487E-20),  # Example: PSF file "0_..."
    "1": (-2.11525294E-20,   4.49097727E-04),  # Example: PSF file "1_..."
    "2": (7.33841152E-20,   8.59391070E-04),
    "4": (3.58508930E-20,   1.20571708E-03),
    "5": (-3.84039716E-20,   1.48153732E-03),
    "7": (1.35464067E-20,   1.69412042E-03),
    "8": (4.98627698E-20,   1.86672617E-03),
    "9": (4.31291668E-21,   2.02867432E-03),
    "11": (-9.25441456E-22,   2.21351601E-03),
    "12": (6.85215234E-20,   2.45295685E-03),
    "14": (-1.23150845E-20,   2.73859288E-03)
}

class StarTrackerPipeline:
    """Star tracker centroiding and bearing vector pipeline simulator."""

    def __init__(self, camera=None, scene_params=None, debug=False):
        """
        Initialize the star tracker pipeline.

        Args:
            camera: Star camera model (if None, default will be created)
            scene_params: Scene parameters (if None, default will be created)
            debug: Enable debug logging
        """
        self.debug = debug
        # Set up default camera and scene if not provided
        if camera is None:
            self.camera = self._get_default_camera()
        else:
            self.camera = camera

        if scene_params is None:
            self.scene = self._get_default_scene()
        else:
            self.scene = scene_params

        if debug:
            logger.setLevel(logging.DEBUG)

    def _get_default_camera(self):
        """Create a default star camera model with typical parameters."""
        # Define the optics
        optic = star_camera.optic_stack(
            f_stop=2.0,  # f-stop of the optical system
            ffov=16.0,  # full field of view, degrees
            transmission=0.8,  # transmission efficiency
        )

        # Define the focal plane array
        fpa = star_camera.fpa(
            x_pixels=2048,  # width in pixels
            y_pixels=2048,  # height in pixels
            pitch=5.5,  # width of pixel in um
            qe=0.6,  # quantum efficiency, fraction
            dark_current_ref=125,  # dark current reference, e-/s
            dark_current_ref_temp=25.0,  # dark current reference temperature, °C
            dark_current_coefficient=6.3,  # dark current temperature coefficient, e-/s/°C
            full_well=13500,  # full well capacity, e-
            read_noise=13.0,  # read noise, e-
            bit_depth=12,  # bit depth of the ADC
        )

        # Create the camera model
        camera = star_camera(
            optic=optic,
            fpa=fpa,
            passband=[0.4, 0.8],  # Passband [min, max] in um
            psf_mult=1.0,  # PSF multiplier
        )

        return camera

    def _get_default_scene(self):
        """Create a default observation scene."""
        return scene(
            int_time=17.0,  # integration time, ms
            temp=20.0,  # temperature, °C
            slew_rate=0.1,  # slew rate, °/s
            fwhm=2.0,  # full width at half maximum, px
        )

    def _get_psf_data_spacing_microns(self, psf_metadata):
        """Helper to get PSF data spacing in microns from metadata."""
        default_spacing_um = 0.550  # Default if not found or invalid

        if psf_metadata and 'data_spacing' in psf_metadata:
            data_spacing_value = psf_metadata['data_spacing']
            if isinstance(data_spacing_value, (float, int)):
                spacing_um = float(data_spacing_value)
                if spacing_um > 0:
                    logger.debug(f"Using PSF 'data_spacing': {spacing_um} µm.")
                    return spacing_um
                else:
                    logger.warning(f"Invalid 'data_spacing' value ({spacing_um} µm) in PSF metadata (must be > 0). Using default {default_spacing_um} µm.")
                    return default_spacing_um
            # The string parsing part is removed as parse_psf_file now handles conversion to float.
            else:
                logger.warning(f"Unexpected type for 'data_spacing' in PSF metadata: {type(data_spacing_value)}. Expected float. Using default {default_spacing_um} µm.")
                return default_spacing_um
        else:
            logger.info(f"PSF metadata does not contain 'data_spacing' or metadata is None. Using default {default_spacing_um} µm.")
            return default_spacing_um

    def update_optical_parameters(self, f_stop=None, aperture=None, focal_length=None):
        """
        Update optical parameters while maintaining physical constraints.
        Only provide one or two parameters, and the third will be calculated.

        Args:
            f_stop: New f-stop value
            aperture: New aperture diameter in mm
            focal_length: New focal length in mm

        Returns:
            dict: Updated optical parameters
        """
        # Get current values
        # current_f_stop = self.camera.optic.f_stop # Not used
        # current_aperture = self.camera.aperature # Not used
        # current_focal_length = self.camera.f_length # Not used

        # Check which parameters were provided and update accordingly
        if f_stop is not None and aperture is not None:
            # Calculate focal length from f-stop and aperture
            new_focal_length = f_stop * aperture

            # Update values
            self.camera.optic.f_stop = f_stop
            self.camera.aperature = aperture
            self.camera.f_length = new_focal_length

            # Update FOV based on new focal length
            self._update_fov_from_focal_length(new_focal_length)

        elif f_stop is not None and focal_length is not None:
            # Calculate aperture from f-stop and focal length
            new_aperture = focal_length / f_stop

            # Update values
            self.camera.optic.f_stop = f_stop
            self.camera.aperature = new_aperture
            self.camera.f_length = focal_length

            # Update FOV based on new focal length
            self._update_fov_from_focal_length(focal_length)

        elif aperture is not None and focal_length is not None:
            # Calculate f-stop from aperture and focal length
            new_f_stop = focal_length / aperture

            # Update values
            self.camera.optic.f_stop = new_f_stop
            self.camera.aperature = aperture
            self.camera.f_length = focal_length

            # Update FOV based on new focal length
            self._update_fov_from_focal_length(focal_length)

        elif f_stop is not None:
            # Keep focal length constant, update aperture
            new_aperture = self.camera.f_length / f_stop

            # Update values
            self.camera.optic.f_stop = f_stop
            self.camera.aperature = new_aperture

        elif aperture is not None:
            # Keep focal length constant, update f-stop
            new_f_stop = self.camera.f_length / aperture

            # Update values
            self.camera.optic.f_stop = new_f_stop
            self.camera.aperature = aperture

        elif focal_length is not None:
            # Keep f-stop constant, update aperture
            new_aperture = focal_length / self.camera.optic.f_stop

            # Update values
            self.camera.aperature = new_aperture
            self.camera.f_length = focal_length

            # Update FOV based on new focal length
            self._update_fov_from_focal_length(focal_length)

        # Recalculate aperture area
        self.camera.aperature_area = np.pi * (self.camera.aperature / 2) ** 2

        # Recalculate airy disc and PSF diameter
        self.camera.airy_disc = (
            2.44
            * self.camera.cent_wavelength
            * self.camera.optic.f_stop
            / self.camera.fpa.pitch
        )
        self.camera.psf_d = self.camera.airy_disc * self.camera.psf_mult

        # Recalculate pixel angular resolution
        self.camera.pixel_angular_resolution = (
            np.degrees(np.arctan((self.camera.fpa.pitch * 1e-3) / self.camera.f_length))
            * 3600
        )

        # Validate constraints
        self._validate_optical_constraints()

        # Return updated parameters
        return {
            "f_stop": self.camera.optic.f_stop,
            "aperture": self.camera.aperature,
            "focal_length": self.camera.f_length,
            "fov": self.camera.optic.ffov,
            "pixel_angular_resolution": self.camera.pixel_angular_resolution,
        }
    
    def load_psf_data(self, psf_directory, pattern="*.txt", apply_pixel_center_offset=False):
        """
        Load PSF data from files in the specified directory.
        
        Args:
            psf_directory: Directory containing PSF files
            pattern: Glob pattern to match PSF files
            apply_pixel_center_offset: If True, shift PSF center from pixel corner to pixel center
                                     by applying a (0.5, 0.5) pixel offset using scipy.ndimage.shift
            
        Returns:
            dict: Dictionary mapping field angles to PSF data
        """
        psf_files = glob.glob(os.path.join(psf_directory, pattern))
        if not psf_files:
            logger.warning(f"No PSF files found in {psf_directory} matching {pattern}")
            return {}
        
        logger.info(f"Found {len(psf_files)} PSF files")
        
        psf_data = {}
        for file_path in psf_files:
            try:
                # Use the existing parse_psf_file function
                metadata, intensity_data = parse_psf_file(file_path)
                
                # Apply pixel center offset if requested
                if apply_pixel_center_offset:
                    # Shift PSF center from pixel corner to pixel center
                    # This moves the PSF center by (0.5, 0.5) pixels to simulate worst-case centroiding
                    original_sum = np.sum(intensity_data)
                    intensity_data = ndimage.shift(intensity_data, (0.5, 0.5), mode='constant', cval=0.0)
                    
                    # Preserve total intensity after shift (slight loss due to boundary effects)
                    shifted_sum = np.sum(intensity_data)
                    if shifted_sum > 0:
                        intensity_data = intensity_data * (original_sum / shifted_sum)
                    
                    logger.info(f"Applied pixel center offset to {os.path.basename(file_path)} "
                              f"(intensity conservation: {np.sum(intensity_data)/original_sum:.4f})")
                
                field_angle = metadata.get('field_angle', None)
                
                if field_angle is None:
                    # Try to extract from filename
                    import re
                    filename = os.path.basename(file_path)
                    # Updated regex for "X_deg.txt" format
                    match = re.search(r"([\d\.]+)_deg\.txt", filename)
                    if match:
                        try:
                            angle_str = match.group(1)
                            field_angle = float(angle_str)
                            logger.info(f"Extracted field angle {field_angle}° from filename {filename}")
                        except ValueError:
                            logger.warning(f"Could not convert extracted angle '{angle_str}' to float from filename {filename}, skipping file.")
                            continue # Skip this file
                    else:
                        logger.warning(f"Could not determine field angle for {file_path} from metadata or filename, skipping")
                        continue
                
                psf_data[field_angle] = {
                    'metadata': metadata,
                    'intensity_data': intensity_data,
                    'file_path': file_path
                }
                
                logger.info(f"Loaded PSF data for field angle {field_angle}°")
                
            except Exception as e:
                logger.warning(f"Error loading PSF file {file_path}: {str(e)}")
        
        return psf_data

    def _update_fov_from_focal_length(self, focal_length):
        """
        Update the field of view based on the new focal length.

        Args:
            focal_length: New focal length in mm
        """
        # Calculate new FOV from focal length and FPA dimensions
        max_dim = max(self.camera.fpa.x_width, self.camera.fpa.y_width)
        fov_half_angle = np.arctan((max_dim / 2) / focal_length)
        new_fov = np.degrees(fov_half_angle) * 2

        # Update FOV parameters
        self.camera.optic.ffov = new_fov
        self.camera.optic.fov_halfang = new_fov / 2

    def _validate_optical_constraints(self):
        """
        Validate physical constraints on optical parameters.
        Raises warnings if constraints are violated.
        """
        # Minimum/maximum f-stop constraints (typical limits)
        if self.camera.optic.f_stop < 0.95:
            logger.warning(
                f"F-stop {self.camera.optic.f_stop:.2f} is below typical minimum (0.95)"
            )
        if self.camera.optic.f_stop > 22:
            logger.warning(
                f"F-stop {self.camera.optic.f_stop:.2f} is above typical maximum (22)"
            )

        # Minimum/maximum aperture constraints (based on common small satellite constraints)
        if self.camera.aperature < 10:
            logger.warning(
                f"Aperture {self.camera.aperature:.2f}mm is very small, expect poor light gathering"
            )
        if self.camera.aperature > 200:
            logger.warning(
                f"Aperture {self.camera.aperature:.2f}mm is very large for a star tracker"
            )

        # FOV constraints for star trackers
        if self.camera.optic.ffov < 5:
            logger.warning(
                f"FOV {self.camera.optic.ffov:.2f}° is very narrow for a star tracker"
            )
        if self.camera.optic.ffov > 40:
            logger.warning(
                f"FOV {self.camera.optic.ffov:.2f}° is very wide for a star tracker"
            )

    def simulate_star(self, magnitude=None, photon_count=None):
        """
        Simulate a star with either a specified magnitude or direct photon count.
        
        Args:
            magnitude: Star magnitude (only used if photon_count is None)
            photon_count: Direct photon count (overrides magnitude if provided)
            
        Returns:
            dict: Star simulation results including photon count
        """
        # Check if we have valid inputs
        if magnitude is None and photon_count is None:
            magnitude = 3.0  # Default to magnitude 3.0
            logger.info(f"No magnitude or photon count specified, using default magnitude {magnitude}")
        
        # Calculate photon count from magnitude if direct count not provided
        if photon_count is None:
            star_obj = star(magnitude=magnitude, passband=self.camera.passband)
            photon_count = calculate_optical_signal(star_obj, self.camera, self.scene)
            logger.info(f"Magnitude {magnitude} star produces {photon_count:.1f} photons")
        else:
            logger.info(f"Using direct photon count: {photon_count}")
        
        return {
            'magnitude': magnitude,
            'photon_count': photon_count
        }

    def project_star_with_psf(self, psf_data, photon_count, num_simulations=1):
        """
        Project a star onto a simulated image using PSF data and Poisson noise.
        
        Args:
            psf_data: PSF data from load_psf_data (a dict containing intensity_data, metadata, file_path)
            photon_count: Number of photons from the star
            num_simulations: Number of Monte Carlo simulations to run
            
        Returns:
            dict: Simulation results including synthetic images
        """
        # Ensure PSF is properly normalized
        psf_intensity = psf_data['intensity_data']
        if np.sum(psf_intensity) > 0:
            normalized_psf = psf_intensity / np.sum(psf_intensity)
        else:
            logger.warning("PSF contains only zeros or is empty!")
            normalized_psf = psf_intensity # Should be an array of zeros of same shape
        
        # Simulate photon distribution with Poisson noise
        logger.info(f"Simulating star with {photon_count:.1f} photons")
        simulation_results = simulate_psf_with_poisson_noise(
            normalized_psf, photon_count, num_simulations=num_simulations
        )
        
        # Add background noise to images
        for i in range(len(simulation_results['simulations'])):
            # Convert to float before adding noise
            simulation_results['simulations'][i] = simulation_results['simulations'][i].astype(np.float64)
            
            # Add background noise
            background_noise_level = 3 # Typical background noise std deviation
            background_noise = np.random.normal(0, background_noise_level, simulation_results['simulations'][i].shape)
            simulation_results['simulations'][i] += background_noise
            # Ensure no negative pixel values after adding noise
            simulation_results['simulations'][i] = np.clip(simulation_results['simulations'][i], 0, None)
        
        return simulation_results

    def calculate_true_psf_centroid(self, psf_item, use_zemax_offsets=False):
        """
        Calculate the true centroid of a PSF.
        If use_zemax_offsets is True and the PSF filename matches a key in `centroid_offsets`, uses the offset.
        Otherwise, calculates from its intensity data using the moment method.
        
        Args:
            psf_item: dict, containing 'intensity_data', 'metadata', and 'file_path'
            use_zemax_offsets: If True, attempt to use Zemax-provided offsets (default False due to reliability issues)
            
        Returns:
            tuple: (x_centroid_px, y_centroid_px) of the PSF in PSF array coordinates
        """
        psf_intensity_data = psf_item['intensity_data']
        psf_metadata = psf_item['metadata']
        psf_file_path = psf_item['file_path']
        filename = os.path.basename(psf_file_path)

        height, width = psf_intensity_data.shape
        geometric_center_x_px = (width - 1) / 2.0
        geometric_center_y_px = (height - 1) / 2.0

        # Check if this is FPA-projected data
        is_fpa_projected = psf_metadata.get('fpa_projected', False) if psf_metadata else False
        
        # Only try to use the centroid_offsets dictionary if explicitly requested and not FPA-projected
        if use_zemax_offsets and not is_fpa_projected:
            for key_str, offset_mm in centroid_offsets.items():
                if filename.startswith(str(key_str)):
                    psf_data_spacing_um = self._get_psf_data_spacing_microns(psf_metadata)
                    if psf_data_spacing_um is None or psf_data_spacing_um == 0:
                        logger.warning(f"Cannot use offset for {filename}: invalid PSF data spacing ({psf_data_spacing_um} µm). Falling back to moment method.")
                        break # Break from loop and fall through to moment method

                    offset_x_mm, offset_y_mm = offset_mm
                    # Convert mm offset to pixel offset
                    # Offset is relative to the optical axis, which should correspond to the geometric center of the PSF array
                    offset_x_px = (offset_x_mm * 1000) / psf_data_spacing_um # (mm * um/mm) / (um/px) = px
                    offset_y_px = (offset_y_mm * 1000) / psf_data_spacing_um # (mm * um/mm) / (um/px) = px

                    # The true center is the geometric center of the PSF array plus the offset.
                    # The PSF array origin (0,0) is typically top-left.
                    # X increases to the right, Y increases downwards.
                    true_x_px = geometric_center_x_px + offset_x_px
                    true_y_px = geometric_center_y_px + offset_y_px # Assuming positive y in offset_mm means downwards in PSF array
                    
                    logger.info(f"Using pre-defined offset for {filename}: ({offset_x_mm:.3e}, {offset_y_mm:.3e}) mm -> ({offset_x_px:.3f}, {offset_y_px:.3f}) px. Geometric center: ({geometric_center_x_px:.3f}, {geometric_center_y_px:.3f}) px. True center: ({true_x_px:.3f}, {true_y_px:.3f}) px")
                    return true_x_px, true_y_px
        
        # Use moment-based calculation (default behavior due to Zemax offset reliability issues)
        if is_fpa_projected:
            logger.info(f"Calculating centroid using moment method for FPA-projected {filename}.")
        else:
            logger.info(f"Calculating centroid using moment method for {filename}.")
            
        y_indices, x_indices = np.indices(psf_intensity_data.shape)
        total_intensity = np.sum(psf_intensity_data)
        
        if total_intensity <= 0:
            logger.warning(f"PSF {filename} has no intensity. Returning geometric center ({geometric_center_x_px}, {geometric_center_y_px}).")
            return geometric_center_x_px, geometric_center_y_px
        
        x_centroid = np.sum(x_indices * psf_intensity_data) / total_intensity
        y_centroid = np.sum(y_indices * psf_intensity_data) / total_intensity
        
        # Log the appropriate scale information
        if is_fpa_projected:
            fpa_pixel_spacing = psf_metadata.get('data_spacing_um', 5.5)
            logger.info(f"Calculated moment-based centroid for FPA-projected {filename}: ({x_centroid:.3f}, {y_centroid:.3f}) px at {fpa_pixel_spacing} µm/px scale")
        else:
            logger.info(f"Calculated moment-based centroid for {filename}: ({x_centroid:.3f}, {y_centroid:.3f}) px.")
            
        return x_centroid, y_centroid
    
    def calculate_bearing_vector_from_field_angle(self, field_angle_deg):
        """
        Calculate the true bearing vector for a given field angle.
        
        Args:
            field_angle_deg: Field angle in degrees
            
        Returns:
            np.array: Unit bearing vector pointing to the star
        """
        # Convert field angle to radians
        field_angle_rad = np.radians(field_angle_deg)
        
        # For simplicity, assume the field angle is radially from boresight
        # The bearing vector in 3D space would be:
        # z = cos(field_angle)
        # r = sin(field_angle) where r is the radial distance in the x-y plane
        # We'll put it along the x-axis for consistency
        
        x = np.sin(field_angle_rad)
        y = 0.0
        z = np.cos(field_angle_rad)
        
        # Create unit vector
        vector = np.array([x, y, z])
        
        # Normalize (should already be unit, but ensure numerical precision)
        return vector / np.linalg.norm(vector)
    
    def calculate_pixel_position_from_field_angle(self, field_angle_deg, image_shape):
        """
        Calculate the expected pixel position for a star at a given field angle.
        
        Args:
            field_angle_deg: Field angle in degrees
            image_shape: Tuple (height, width) of the image
            
        Returns:
            tuple: (x_pixel, y_pixel) where the star should appear
        """
        # Get image dimensions
        height, width = image_shape
        center_x, center_y = width / 2.0, height / 2.0
        
        # Calculate focal length in pixels
        focal_length_px = self.camera.f_length * 1000 / self.camera.fpa.pitch
        
        # Convert field angle to radians
        field_angle_rad = np.radians(field_angle_deg)
        
        # For a star at field angle θ from boresight (along x-axis for simplicity):
        # The star appears at a distance from center = focal_length * tan(θ)
        # This is the pinhole camera model
        distance_from_center_px = focal_length_px * np.tan(field_angle_rad)
        
        # Place it along the x-axis (positive x direction)
        x_pixel = center_x + distance_from_center_px
        y_pixel = center_y  # No y-offset for stars along x-axis
        
        return x_pixel, y_pixel

    def detect_stars_and_calculate_centroids(self, images, k_sigma=5.0, min_pixels=5, max_pixels=100, block_size=32, true_center=None, return_debug_data=False):
        """
        Detect stars in simulated images and calculate their centroids using adaptive local thresholding.
        
        Args:
            images: List of synthetic star images
            k_sigma: Number of standard deviations above mean for threshold
            min_pixels: Minimum pixel count for valid star region
            max_pixels: Maximum pixel count for valid star region
            block_size: Size of blocks for local statistics in adaptive thresholding
            true_center: The true center position to compare against (if None, will use image center)
            return_debug_data: If True, return intermediate data for debugging (default False)
            
        Returns:
            dict: Detection and centroiding results
            (optional) dict: Intermediate debug data if return_debug_data is True
        """
        results = {
            'successful_detections': 0,
            'centroids': [],
            'centroid_errors': [],
            'true_center': None
        }
        
        debug_data = {
            'threshold_maps': [],
            'binary_images': [],
            'labels': [],
            'stats': [],
            'valid_regions': []
        } if return_debug_data else None

        if not images: # Check if the list of images is empty
            logger.warning("No images provided for star detection.")
            return (results, debug_data) if return_debug_data else results

        height, width = images[0].shape
        
        # Use provided true center or default to image center
        if true_center is None:
            true_center = (width / 2.0, height / 2.0)
            logger.warning("No true center provided, using image center as reference")
        
        results['true_center'] = true_center
        
        for image_idx, image in enumerate(images):
            try:
                # --- Adaptive Local Thresholding ---
                num_blocks_y = height // block_size
                num_blocks_x = width // block_size

                # Handle cases where image is smaller than block_size
                if num_blocks_y == 0: num_blocks_y = 1
                if num_blocks_x == 0: num_blocks_x = 1

                local_mean_map = np.zeros((num_blocks_y, num_blocks_x))
                local_std_map = np.zeros((num_blocks_y, num_blocks_x))

                for i in range(num_blocks_y):
                    for j in range(num_blocks_x):
                        # Define block boundaries
                        y_start, y_end = i * block_size, min((i + 1) * block_size, height)
                        x_start, x_end = j * block_size, min((j + 1) * block_size, width)
                        block = image[y_start:y_end, x_start:x_end]
                        
                        if block.size > 0:
                            local_mean_map[i, j] = np.mean(block)
                            local_std_map[i, j] = np.std(block)
                        else: # Should not happen if image is not empty and block_size is reasonable
                            local_mean_map[i, j] = 0
                            local_std_map[i, j] = 0
                
                # Upsample local statistics to full image size
                # cv2.INTER_NEAREST ensures that each pixel within a block gets the block's threshold
                mean_upsampled = cv2.resize(local_mean_map, (width, height), interpolation=cv2.INTER_NEAREST)
                std_upsampled = cv2.resize(local_std_map, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Calculate adaptive threshold map
                threshold_map = mean_upsampled + k_sigma * std_upsampled
                
                # Create binary mask using the adaptive threshold map
                binary_image = (image > threshold_map).astype(np.uint8)
                # --- End of Adaptive Local Thresholding ---
                
                # Connected components analysis (from identify.group_pixels, but cv2 is directly used here)
                num_labels, labels, stats, cc_centroids = cv2.connectedComponentsWithStats(
                    binary_image, connectivity=8 # 8-connectivity is standard
                )
                
                if return_debug_data:
                    debug_data['threshold_maps'].append(threshold_map)
                    debug_data['binary_images'].append(binary_image)
                    debug_data['labels'].append(labels)
                    debug_data['stats'].append(stats)

                if num_labels <= 1:  # Only background
                    if self.debug:
                        logger.debug(f"Image {image_idx}: No components found after adaptive thresholding.")
                    if return_debug_data: debug_data['valid_regions'].append([])
                    continue
                    
                # Get valid regions (filter by size)
                valid_regions_indices = []
                for i in range(1, num_labels): # Skip background label 0
                    area = stats[i, cv2.CC_STAT_AREA]
                    if min_pixels <= area <= max_pixels:
                        valid_regions_indices.append(i)
                
                if return_debug_data:
                    debug_data['valid_regions'].append(valid_regions_indices)

                if not valid_regions_indices:
                    if self.debug:
                        logger.debug(f"Image {image_idx}: No regions met size criteria ({min_pixels}-{max_pixels} pixels).")
                    continue
                
                # IMPROVED REGION SELECTION: Prioritize brightest region instead of closest to true_center
                if len(valid_regions_indices) > 1:
                    # Calculate total intensity for each valid region
                    region_intensities = []
                    for r_idx in valid_regions_indices:
                        region_mask = (labels == r_idx)
                        total_intensity = np.sum(image * region_mask)
                        region_intensities.append(total_intensity)
                    
                    # Select region with highest total intensity
                    brightest_idx = np.argmax(region_intensities)
                    closest_region_idx = valid_regions_indices[brightest_idx]
                    
                    if self.debug:
                        logger.debug(f"Image {image_idx}: Selected brightest region (index {closest_region_idx}) "
                                   f"with intensity {region_intensities[brightest_idx]:.1f} "
                                   f"from {len(valid_regions_indices)} candidates")
                else:
                    closest_region_idx = valid_regions_indices[0]
                
                # Create mask for the selected star region and calculate its centroid
                star_region_mask = (labels == closest_region_idx)
                # Pass the original image (not thresholded) for intensity weighting
                centroid_result = calculate_centroid(star_region_mask, image)
                
                if centroid_result is None:
                    if self.debug:
                        logger.debug(f"Image {image_idx}: Centroid calculation failed for region {closest_region_idx}.")
                    continue
                    
                x_centroid, y_centroid, _ = centroid_result # Third element is total_intensity
                centroid_error = np.sqrt((x_centroid - true_center[0])**2 + (y_centroid - true_center[1])**2)
                
                results['centroids'].append((x_centroid, y_centroid))
                results['centroid_errors'].append(centroid_error)
                results['successful_detections'] += 1
                
            except Exception as e:
                if self.debug:
                    logger.debug(f"Image {image_idx}: Error in star detection/centroiding: {str(e)}")
                # import traceback # Optional: for more detailed error
                # logger.debug(traceback.format_exc()) # Optional
                continue
        
        # Calculate statistics
        if results['successful_detections'] > 0:
            results['success_rate'] = results['successful_detections'] / len(images)
            results['mean_error'] = np.mean(results['centroid_errors'])
            results['std_error'] = np.std(results['centroid_errors'])
        else:
            results['success_rate'] = 0.0
            results['mean_error'] = float('nan')
            results['std_error'] = float('nan')
        
        return (results, debug_data) if return_debug_data else results

    def calculate_bearing_vectors(self, centroids, image_shape, sensor_pixel_pitch_um, true_bearing_vector=None): # Added sensor_pixel_pitch_um, removed focal_length_px
        """
        Convert pixel centroids to bearing vectors.
        
        Args:
            centroids: List of (x, y) centroid coordinates
            image_shape: Tuple (height, width) of the image from which centroids were derived.
            sensor_pixel_pitch_um: The pixel pitch (in microns) of the sensor plane for these centroids.
            true_bearing_vector: The expected bearing vector for error calculation (if None, uses boresight)
            
        Returns:
            dict: Bearing vector calculation results
        """
        # Calculate focal length in pixels using the provided sensor_pixel_pitch_um
        if sensor_pixel_pitch_um is None or sensor_pixel_pitch_um <= 0:
            logger.error(f"Invalid sensor_pixel_pitch_um ({sensor_pixel_pitch_um} µm) provided. Cannot calculate bearing vectors.")
            return { 
                'bearing_vectors': [], 'vector_errors_arcsec': [],
                'mean_vector_error_arcsec': float('nan'), 'std_vector_error_arcsec': float('nan')
            }
        focal_length_px = self.camera.f_length * 1000 / sensor_pixel_pitch_um # mm * um/mm / um/px = px
        
        # Get image center based on the actual image dimensions used for centroiding
        height, width = image_shape 
        center_x, center_y = width / 2.0, height / 2.0 # Use float for center
        
        bearing_vectors = []
        vector_errors_arcsec = []
        
        # Reference vector - use provided true bearing vector or default to boresight
        if true_bearing_vector is not None:
            ref_vector = true_bearing_vector
        else:
            ref_vector = np.array([0.0, 0.0, 1.0]) # Standard convention: Z is optical axis, X,Y in focal plane
        
        for x, y in centroids:
            # Convert to focal plane coordinates relative to the image center
            # These are displacements in the sensor plane (e.g., in units of pixels if focal_length_px is used for normalization)
            # If (0,0) is top-left of sensor:
            # x_sensor = (x - center_x) * self.camera.fpa.pitch  (in um)
            # y_sensor = (center_y - y) * self.camera.fpa.pitch  (in um, y is often inverted in image coords)
            # Let's use the simpler (x - center_x) / focal_length_px convention for now,
            # assuming x,y are pixel coordinates and image center is (center_x, center_y)
            
            X_focal_plane = (x - center_x) / focal_length_px 
            Y_focal_plane = (y - center_y) / focal_length_px 
                        # Sign convention might need checking depending on image y-axis
                        # If image y-axis is downwards, this Y_focal_plane might need negation
                        # for a standard right-handed camera coordinate system.
                        # For now, assume y increases downwards in image, upwards in camera frame.
                        # So a positive (y - center_y) means below center, which is -Y in camera frame.
                        # Let's assume (x - center_x) maps to camera X, and (y - center_y) maps to camera Y for now.
                        # This was the previous convention: vector = [1.0, X_fp, Y_fp] implying Z was [1.0]
                        # If Z is optical axis, then vector is [X_fp, Y_fp, 1.0] before normalization.

            # Vector in camera frame: [x_cam, y_cam, z_cam]
            # x_cam = X_focal_plane
            # y_cam = Y_focal_plane (sign might need adjustment based on image vs camera y-axis)
            # z_cam = 1.0 (arbitrary distance along optical axis, will be normalized)
            vector = np.array([X_focal_plane, Y_focal_plane, 1.0]) 
            
            # Normalize to unit vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                 vector = vector / norm
            else: 
                logger.warning("Bearing vector has zero norm before normalization.")
                vector = np.array([0.0, 0.0, 1.0]) # Default to boresight along Z

            # Calculate angular error from reference (in radians)
            dot_product = np.clip(np.dot(ref_vector, vector), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_arcsec = np.degrees(angle_rad) * 3600
            
            bearing_vectors.append(vector)
            vector_errors_arcsec.append(angle_arcsec)
        
        # Calculate statistics
        results = {
            'bearing_vectors': bearing_vectors,
            'vector_errors_arcsec': vector_errors_arcsec
        }
        
        if bearing_vectors:
            results['mean_vector_error_arcsec'] = np.mean(vector_errors_arcsec)
            results['std_vector_error_arcsec'] = np.std(vector_errors_arcsec)
        else:
            results['mean_vector_error_arcsec'] = float('nan')
            results['std_vector_error_arcsec'] = float('nan')
        
        return results

    def run_monte_carlo_simulation(self, psf_data, magnitude=None, photon_count=None, 
                                   num_trials=100, threshold_sigma=4.0, adaptive_block_size=32):
        """
        Run Monte Carlo simulation of star detection and centroiding.
        
        Args:
            psf_data: PSF data (dictionary with metadata, intensity_data, and file_path)
            magnitude: Star magnitude (only used if photon_count is None)
            photon_count: Direct photon count (overrides magnitude if provided)
            num_trials: Number of Monte Carlo trials
            threshold_sigma: Sigma multiplier for detection threshold
            adaptive_block_size: Block size for adaptive thresholding
            
        Returns:
            dict: Monte Carlo simulation results
        """
        # --- BEGIN DEBUG PRINT --- 
        if self.debug and psf_data and 'metadata' in psf_data and psf_data['metadata']:
            ds_val = psf_data['metadata'].get('data_spacing')
            ds_type = type(ds_val)
            logger.info(f"DEBUG run_monte_carlo_simulation: Received metadata['data_spacing'] = {ds_val}, type: {ds_type} for file {psf_data.get('file_path')}")
        elif self.debug:
            logger.info(f"DEBUG run_monte_carlo_simulation: psf_data or metadata missing/empty for file {psf_data.get('file_path')}")
        # --- END DEBUG PRINT ---

        # Simulate star
        star_simulation = self.simulate_star(magnitude, photon_count)
        photon_count_actual = star_simulation['photon_count'] 
        
        # Calculate the true centroid from the original PSF
        true_centroid = self.calculate_true_psf_centroid(psf_data)
        
        # Project star using PSF
        projection_results = self.project_star_with_psf(
            psf_data, photon_count_actual, num_simulations=num_trials 
        )
        
        sim_image_shape = None
        if projection_results['simulations'] and len(projection_results['simulations']) > 0:
            sim_image_shape = projection_results['simulations'][0].shape
        # Else, sim_image_shape remains None. This is handled before calling calculate_bearing_vectors.

        centroid_results = self.detect_stars_and_calculate_centroids(
            projection_results['simulations'], threshold_sigma, block_size=adaptive_block_size,
            true_center=true_centroid 
        )
        
        # Initialize bearing_results to ensure it's always defined
        bearing_results = { 
            'bearing_vectors': [], 'vector_errors_arcsec': [],
            'mean_vector_error_arcsec': float('nan'), 'std_vector_error_arcsec': float('nan')
        }

        if centroid_results['successful_detections'] > 0:
            if sim_image_shape is None:
                logger.error("sim_image_shape is None but detections were successful. Cannot calculate bearing vectors for original PSF.")
            else:
                # Determine the pixel pitch for the original PSF grid
                original_psf_pixel_pitch_um = self._get_psf_data_spacing_microns(psf_data.get('metadata'))
                if original_psf_pixel_pitch_um is None or original_psf_pixel_pitch_um <= 0:
                    logger.error(f"Invalid data spacing ({original_psf_pixel_pitch_um}) for original PSF. Bearing vector calculation will be incorrect.")
                    # Fallback to a default or handle error, here we use a clearly incorrect value for focal_length_px to highlight issue if it proceeds
                    original_psf_pixel_pitch_um = 1.0 # Avoid division by zero, but this is a placeholder if parsing failed badly

                # Calculate the true bearing vector from the actual PSF centroid position on its own grid scale
                true_bearing_vector_results = self.calculate_bearing_vectors(
                    [true_centroid],  # The true centroid position in original PSF pixels
                    image_shape=sim_image_shape,
                    sensor_pixel_pitch_um=original_psf_pixel_pitch_um, # Use original PSF's pixel pitch
                    true_bearing_vector=None  # Don't compare to anything yet, this IS the reference
                )
                
                if true_bearing_vector_results['bearing_vectors']:
                    true_bearing_vector = true_bearing_vector_results['bearing_vectors'][0]
                else:
                    logger.warning("Failed to calculate true_bearing_vector for original PSF. Defaulting to boresight.")
                    true_bearing_vector = np.array([0.0, 0.0, 1.0]) # Fallback to boresight
                
                # Now calculate bearing vectors for detected centroids (on original PSF grid) and compare to true
                bearing_results = self.calculate_bearing_vectors(
                    centroid_results['centroids'], 
                    image_shape=sim_image_shape,
                    sensor_pixel_pitch_um=original_psf_pixel_pitch_um, # Use original PSF's pixel pitch
                    true_bearing_vector=true_bearing_vector
                )
        
        # Print immediate status
        if centroid_results['successful_detections'] > 0:
            logger.info(f"    Successful detections: {centroid_results['successful_detections']}/{num_trials}")
            logger.info(f"    Mean centroid error: {centroid_results['mean_error']:.3f} pixels")
            if 'mean_vector_error_arcsec' in bearing_results and not np.isnan(bearing_results['mean_vector_error_arcsec']):
                logger.info(f"    Mean vector error: {bearing_results['mean_vector_error_arcsec']:.2f} arcsec")
        else:
            logger.warning(f"    No successful detections for this field angle")

        # Get PSF data spacing for micron conversion
        psf_pixel_spacing_um = self._get_psf_data_spacing_microns(psf_data.get('metadata'))
        
        mean_centroid_error_um_val = float('nan')
        std_centroid_error_um_val = float('nan')
        if not np.isnan(centroid_results['mean_error']):
            mean_centroid_error_um_val = centroid_results['mean_error'] * psf_pixel_spacing_um
        if not np.isnan(centroid_results['std_error']):
            std_centroid_error_um_val = centroid_results['std_error'] * psf_pixel_spacing_um

        # Combine all results
        results = {
            'star_simulation': star_simulation,
            'projection_results': projection_results,
            'centroid_results': centroid_results,
            'bearing_results': bearing_results, # This will now be the initialized or calculated version
            'psf_data': psf_data,  # Added to pass original PSF data and metadata
            'successful_trials': centroid_results['successful_detections'],
            'num_trials': num_trials,
            'success_rate': centroid_results['success_rate'],
            'mean_centroid_error_px': centroid_results['mean_error'],
            'std_centroid_error_px': centroid_results['std_error'],
            'mean_centroid_error_um': mean_centroid_error_um_val, # Added
            'std_centroid_error_um': std_centroid_error_um_val,   # Added
            'mean_vector_error_arcsec': bearing_results['mean_vector_error_arcsec'], # From bearing_results
            'std_vector_error_arcsec': bearing_results['std_vector_error_arcsec']   # From bearing_results
        }
        
        # Calculate confidence intervals (95%)
        if centroid_results['successful_detections'] > 2: 
            if not np.isnan(results['std_centroid_error_px']):
                 results['centroid_error_95ci'] = 1.96 * results['std_centroid_error_px'] / np.sqrt(centroid_results['successful_detections'])
            else:
                results['centroid_error_95ci'] = float('nan')

            if not np.isnan(results['std_vector_error_arcsec']): # Check the one from combined results
                results['vector_error_95ci'] = 1.96 * results['std_vector_error_arcsec'] / np.sqrt(centroid_results['successful_detections'])
            else:
                results['vector_error_95ci'] = float('nan')
        else:
            results['centroid_error_95ci'] = float('nan')
            results['vector_error_95ci'] = float('nan')
        
        return results

    def analyze_psf_centroiding_accuracy(self, psf_directory=None, pattern="*_FieldFlattener_*.txt",
                                    num_trials=50, magnitude=None, photon_counts=None, 
                                    output_dir=None, psf_data=None):
        """
        Analyze centroiding accuracy using PSF files.
        
        Args:
            psf_directory: Directory containing PSF files (optional if psf_data provided)
            pattern: Glob pattern to match PSF files
            num_trials: Number of Monte Carlo trials
            magnitude: Star magnitude to simulate
            photon_counts: List of photon counts to simulate (overrides magnitude)
            output_dir: Directory to save output visualizations
            psf_data: Pre-loaded PSF data (optional)
            
        Returns:
            dict: Results of PSF centroiding analysis
        """
        # Use provided PSF data or load from directory
        if psf_data is None and psf_directory is not None:
            psf_data = self.load_psf_data(psf_directory, pattern)
        
        if not psf_data:
            logger.error("No PSF data available for analysis")
            return None
        
        # Define photon counts if not provided
        if photon_counts is None and magnitude is not None:
            # Use a single photon count derived from magnitude
            star_obj = star(magnitude=magnitude, passband=self.camera.passband)
            photon_count_val = calculate_optical_signal(star_obj, self.camera, self.scene) 
            photon_counts = [photon_count_val]
            logger.info(f"Using photon count {photon_count_val:.1f} derived from magnitude {magnitude}")
        elif photon_counts is None:
            # Use default photon counts
            photon_counts = [1000, 3000, 10000, 30000]
            logger.info(f"Using default photon counts: {photon_counts}")
        
        logger.info(f"Analyzing centroiding accuracy for {len(psf_data)} field angles and {len(photon_counts)} photon counts")
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        results = {
            'field_angles': [],
            'centroid_errors': {photons: [] for photons in photon_counts},
            'centroid_stds': {photons: [] for photons in photon_counts},
            'vector_errors_arcsec': {photons: [] for photons in photon_counts},
            'vector_stds_arcsec': {photons: [] for photons in photon_counts},
            'success_rates': {photons: [] for photons in photon_counts},
            'trial_results': {}
        }
        
        # Sort PSF data by field angle for predictable order
        field_angles = sorted(psf_data.keys())
        results['field_angles'] = field_angles
        
        # For each field angle
        for angle in field_angles:
            logger.info(f"Processing field angle {angle}°")
            
            # Get PSF data for this angle
            current_psf_data = psf_data[angle] 
            
            # Initialize per-angle results
            angle_results = {
                'photon_counts': {},
                'success_rates': {},
            }
            
            # Run trials for each photon count
            for photon_count_val in photon_counts: 
                logger.info(f"  Simulating with {photon_count_val} photons")
                
                # Run Monte Carlo simulation using FPA-projected method for realistic CMV4000 simulation
                mc_results = self.run_monte_carlo_simulation_fpa_projected(
                    current_psf_data, photon_count=photon_count_val, num_trials=num_trials,
                    target_pixel_pitch_um=5.5  # CMV4000 pixel pitch
                )
                
                # Store results
                angle_results['photon_counts'][photon_count_val] = mc_results
                angle_results['success_rates'][photon_count_val] = mc_results['success_rate']
                
                # Store in overall results
                results['centroid_errors'][photon_count_val].append(mc_results['mean_centroid_error_px'])
                results['centroid_stds'][photon_count_val].append(mc_results['std_centroid_error_px'])
                results['vector_errors_arcsec'][photon_count_val].append(mc_results['mean_vector_error_arcsec'])
                results['vector_stds_arcsec'][photon_count_val].append(mc_results['std_vector_error_arcsec'])
                results['success_rates'][photon_count_val].append(mc_results['success_rate'])
                
                logger.info(f"    Mean centroid error: {mc_results['mean_centroid_error_px']:.3f} pixels")
                if not np.isnan(mc_results['mean_vector_error_arcsec']):
                    logger.info(f"    Mean bearing vector error: {mc_results['mean_vector_error_arcsec']:.2f} arcsec")
                else:
                    logger.info(f"    Mean bearing vector error: NaN")
                logger.info(f"    Success rate: {mc_results['success_rate']:.2f}")
            
            # Store detailed results for this angle
            results['trial_results'][angle] = angle_results
        
        # Visualize results if output directory specified
        if output_dir:
            self.visualize_psf_analysis_results(results, output_dir)
        
        return results

    def visualize_psf_analysis_results(self, results, output_dir):
        """
        Create visualizations of PSF centroiding analysis results.
        
        Args:
            results: Results dict from analyze_psf_centroiding_accuracy
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        
        field_angles = results['field_angles']
        photon_counts = list(results['centroid_errors'].keys())
        
        # Create color map for different photon counts
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, len(photon_counts)))
        
        # Figure 1: Centroid Error vs Field Angle
        plt.figure(figsize=(10, 6))
        
        for i, photons in enumerate(photon_counts):
            errors = results['centroid_errors'][photons]
            stds = results['centroid_stds'][photons]
            
            mask = ~np.isnan(errors) & ~np.isnan(stds) 
            if not np.any(mask):
                continue
                
            angles_valid = np.array(field_angles)[mask]
            errors_valid = np.array(errors)[mask]
            stds_valid = np.array(stds)[mask]
            
            plt.errorbar(angles_valid, errors_valid, yerr=stds_valid, fmt='o-', 
                        color=colors[i], label=f"{photons} photons", capsize=5)
        
        plt.title('Centroid Error vs Field Angle')
        plt.xlabel('Field Angle (degrees)')
        plt.ylabel('Centroid Error (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "centroid_error_vs_angle.png"), dpi=300)
        plt.close()
        
        # Figure 2: Vector Error vs Field Angle
        plt.figure(figsize=(10, 6))
        
        for i, photons in enumerate(photon_counts):
            errors = results['vector_errors_arcsec'][photons]
            stds = results['vector_stds_arcsec'][photons]
            
            mask = ~np.isnan(errors) & ~np.isnan(stds) 
            if not np.any(mask):
                continue
                
            angles_valid = np.array(field_angles)[mask]
            errors_valid = np.array(errors)[mask]
            stds_valid = np.array(stds)[mask]
            
            plt.errorbar(angles_valid, errors_valid, yerr=stds_valid, fmt='o-', 
                        color=colors[i], label=f"{photons} photons", capsize=5)
        
        plt.title('Bearing Vector Error vs Field Angle')
        plt.xlabel('Field Angle (degrees)')
        plt.ylabel('Angular Error (arcsec)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vector_error_vs_angle.png"), dpi=300)
        plt.close()
        
        # Figure 3: Success Rate vs Field Angle
        plt.figure(figsize=(10, 6))
        
        for i, photons in enumerate(photon_counts):
            rates = results['success_rates'][photons]
            
            mask = ~np.isnan(rates)
            if not np.any(mask):
                continue
                
            angles_valid = np.array(field_angles)[mask]
            rates_valid = np.array(rates)[mask]
            
            plt.plot(angles_valid, rates_valid, 'o-', 
                    color=colors[i], label=f"{photons} photons")
        
        plt.title('Success Rate vs Field Angle')
        plt.xlabel('Field Angle (degrees)')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_rate_vs_angle.png"), dpi=300)
        plt.close()
        
        # Figure 4: Error vs 1/sqrt(photons)
        if len(photon_counts) > 1:
            plt.figure(figsize=(10, 6))
            
            for angle_idx, angle in enumerate(field_angles):
                valid_photons = []
                valid_errors = []
                
                for photons in photon_counts:
                    error = results['vector_errors_arcsec'][photons][angle_idx]
                    if not np.isnan(error):
                        valid_photons.append(photons)
                        valid_errors.append(error)
                
                if len(valid_photons) < 2:
                    continue
                    
                valid_photons_arr = np.array(valid_photons) 
                valid_errors_arr = np.array(valid_errors)   
                
                sort_idx = np.argsort(valid_photons_arr)
                valid_photons_arr = valid_photons_arr[sort_idx]
                valid_errors_arr = valid_errors_arr[sort_idx]
                
                inv_sqrt_photons = 1 / np.sqrt(valid_photons_arr)
                
                plt.plot(inv_sqrt_photons, valid_errors_arr, 'o-', 
                        label=f"Field Angle {angle}°")
                
                if len(valid_photons_arr) > 2: 
                    from scipy import stats
                    finite_mask = np.isfinite(inv_sqrt_photons) & np.isfinite(valid_errors_arr)
                    if np.sum(finite_mask) > 1: 
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            inv_sqrt_photons[finite_mask], valid_errors_arr[finite_mask])
                        
                        x_fit = np.linspace(min(inv_sqrt_photons[finite_mask]), max(inv_sqrt_photons[finite_mask]), 100)
                        y_fit = slope * x_fit + intercept
                        plt.plot(x_fit, y_fit, '--', color='gray', alpha=0.5)
                        
                        logger.info(f"Field Angle {angle}°: Error ~ {slope:.2f} / sqrt(photons), R² = {r_value**2:.3f}")
                    else:
                        logger.warning(f"Field Angle {angle}°: Not enough finite data points for linear regression.")

            plt.title('Vector Error vs 1/sqrt(Photons)')
            plt.xlabel('1/sqrt(Photons)')
            plt.ylabel('Angular Error (arcsec)')
            plt.grid(True, alpha=0.3)
            
            if len(field_angles) <= 10:
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "error_vs_photon_count.png"), dpi=300)
            plt.close()
        
        import pandas as pd
        csv_data = {'Field_Angle': field_angles}
        for photons in photon_counts:
            csv_data[f'Centroid_Error_{photons}'] = results['centroid_errors'][photons]
            csv_data[f'Centroid_Std_{photons}'] = results['centroid_stds'][photons]
            csv_data[f'Vector_Error_{photons}'] = results['vector_errors_arcsec'][photons]
            csv_data[f'Vector_Std_{photons}'] = results['vector_stds_arcsec'][photons]
            csv_data[f'Success_Rate_{photons}'] = results['success_rates'][photons]
        
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(output_dir, "psf_centroiding_results.csv"), index=False)
        logger.info(f"Saved PSF analysis results to {output_dir}")

    def run_optical_parameter_analysis(self, param_name, param_values, magnitude=None, 
                                   photon_count=None, num_trials=30, 
                                   psf_data=None, output_dir=None):
        logger.info(f"Running optical parameter analysis: {param_name} with {len(param_values)} values")
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        if psf_data is None:
            logger.error("PSF data is required for optical parameter analysis")
            return None
        
        field_angles = sorted(psf_data.keys())
        on_axis_idx = np.argmin(np.abs(np.array(field_angles))) 
        test_psf = psf_data[field_angles[on_axis_idx]]
        logger.info(f"Using PSF with field angle {field_angles[on_axis_idx]}° for analysis")
        
        # Store original parameter value correctly
        if param_name == 'f_stop':
            original_value = self.camera.optic.f_stop
        elif param_name == 'focal_length':
            original_value = self.camera.f_length # Correct attribute
        elif param_name == 'pixel_size':
            original_value = self.camera.fpa.pitch # Correct attribute
        else:
            raise ValueError(f"Unknown parameter name for original value: {param_name}")


        analysis_results_list = []
        for value in param_values:
            logger.info(f"Testing {param_name} = {value}")
            param_dir = os.path.join(output_dir, f"{param_name}_{value:.2f}") if output_dir else None
            if param_dir: os.makedirs(param_dir, exist_ok=True)
            
            if param_name == 'f_stop': self.update_optical_parameters(f_stop=value)
            elif param_name == 'focal_length': self.update_optical_parameters(focal_length=value)
            elif param_name == 'pixel_size':
                self.camera.fpa.pitch = value
                self.update_optical_parameters() # Recalculate dependent parameters
            
            mc_results = self.run_monte_carlo_simulation(test_psf, magnitude=magnitude, photon_count=photon_count, num_trials=num_trials)
            
            # Corrected current_params dictionary creation
            current_params = {
                'f_stop': self.camera.optic.f_stop,
                'focal_length': self.camera.f_length,
                'aperture': self.camera.aperature,
                'fov': self.camera.optic.ffov,
                'pixel_size': self.camera.fpa.pitch,
                'pixel_angular_resolution': self.camera.pixel_angular_resolution
            }
            
            result_item = {
                'param_value': value, 'current_params': current_params,
                'centroid_error_px': mc_results['mean_centroid_error_px'], 
                'centroid_error_std': mc_results['std_centroid_error_px'], 
                'vector_error_arcsec': mc_results['mean_vector_error_arcsec'], 
                'vector_error_std': mc_results['std_vector_error_arcsec'], 
                'success_rate': mc_results['success_rate'], 
                'mc_results': mc_results
            }

            analysis_results_list.append(result_item)
            if param_dir: self.visualize_single_parameter_results(mc_results, param_dir)
        
        # Restore original parameter value
        if param_name == 'f_stop': self.update_optical_parameters(f_stop=original_value)
        elif param_name == 'focal_length': self.update_optical_parameters(focal_length=original_value)
        elif param_name == 'pixel_size':
            self.camera.fpa.pitch = original_value
            self.update_optical_parameters() # Recalculate
            
        if output_dir and analysis_results_list:
            self.visualize_parameter_analysis_results(param_name, analysis_results_list, output_dir)
        return {'param_name': param_name, 'results': analysis_results_list}

    def visualize_single_parameter_results(self, mc_results, output_dir):
        import matplotlib.pyplot as plt
        if mc_results['projection_results'] and mc_results['projection_results']['simulations']:
            plt.figure(figsize=(8, 8))
            image = mc_results['projection_results']['simulations'][0]
            plt.imshow(image, cmap='viridis', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title('Example Simulated Star Image')
            if mc_results['centroid_results'] and mc_results['centroid_results']['true_center']:
                true_center = mc_results['centroid_results']['true_center']
                plt.plot(true_center[0], true_center[1], 'rx', markersize=10, label='True Center')
            if mc_results['centroid_results'] and mc_results['centroid_results']['centroids']:
                for x, y in mc_results['centroid_results']['centroids']: plt.plot(x, y, 'g+', markersize=8)
            plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "example_image.png"), dpi=300); plt.close()

        if mc_results['centroid_results'] and mc_results['centroid_results']['centroid_errors']:
            plt.figure(figsize=(8, 6))
            errors = mc_results['centroid_results']['centroid_errors']
            plt.hist(errors, bins=20, color='blue', alpha=0.7)
            if not np.isnan(mc_results['mean_centroid_error_px']):
                plt.axvline(mc_results['mean_centroid_error_px'], color='red', linestyle='--', label=f"Mean: {mc_results['mean_centroid_error_px']:.3f} px")
            plt.title('Centroid Error Distribution'); plt.xlabel('Error (pixels)'); plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "centroid_error_histogram.png"), dpi=300); plt.close()

        if mc_results['bearing_results'] and mc_results['bearing_results']['vector_errors_arcsec']:
            plt.figure(figsize=(8, 6))
            errors = mc_results['bearing_results']['vector_errors_arcsec']
            plt.hist(errors, bins=20, color='green', alpha=0.7)
            if not np.isnan(mc_results['mean_vector_error_arcsec']):
                plt.axvline(mc_results['mean_vector_error_arcsec'], color='red', linestyle='--', label=f"Mean: {mc_results['mean_vector_error_arcsec']:.2f} arcsec")
            plt.title('Bearing Vector Error Distribution'); plt.xlabel('Error (arcsec)'); plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "vector_error_histogram.png"), dpi=300); plt.close()

    def visualize_parameter_analysis_results(self, param_name, analysis_results_list, output_dir):
        import matplotlib.pyplot as plt
        param_values = [r['param_value'] for r in analysis_results_list]
        centroid_errors = [r['centroid_error_px'] for r in analysis_results_list]
        centroid_stds = [r['centroid_error_std'] for r in analysis_results_list] 
        vector_errors = [r['vector_error_arcsec'] for r in analysis_results_list]
        vector_stds = [r['vector_error_std'] for r in analysis_results_list] 
        success_rates = [r['success_rate'] for r in analysis_results_list]

        plots = [
            ('Centroid Error', 'Centroid Error (pixels)', centroid_errors, centroid_stds, 'blue'),
            ('Bearing Vector Error', 'Angular Error (arcsec)', vector_errors, vector_stds, 'green'),
            ('Success Rate', 'Success Rate', success_rates, None, 'purple')
        ]
        for title, ylabel, data, stds, color in plots:
            plt.figure(figsize=(10, 6))
            if stds: plt.errorbar(param_values, data, yerr=stds, fmt='o-', color=color, capsize=5)
            else: plt.plot(param_values, data, 'o-', color=color)
            plt.title(f'{title} vs {param_name.capitalize()}'); plt.xlabel(param_name.capitalize()); plt.ylabel(ylabel)
            if title == 'Success Rate': plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_vs_{param_name}.png"), dpi=300); plt.close()

        if param_name == 'focal_length':
            fovs = [r['current_params']['fov'] for r in analysis_results_list]
            pixel_resolutions = [r['current_params']['pixel_angular_resolution'] for r in analysis_results_list]
            extra_plots = [
                ('Field of View vs Focal Length', 'Focal Length (mm)', 'Field of View (degrees)', fovs, 'orange'),
                ('Pixel Angular Resolution vs Focal Length', 'Focal Length (mm)', 'Pixel Resolution (arcsec/pixel)', pixel_resolutions, 'red'),
                ('Bearing Vector Error vs Pixel Resolution', 'Pixel Resolution (arcsec/pixel)', 'Angular Error (arcsec)', vector_errors, 'cyan', pixel_resolutions) # x_data is pixel_resolutions
            ]
            for title, xlabel, ylabel, y_data, color, *x_data_arg in extra_plots:
                x_data = x_data_arg[0] if x_data_arg else param_values
                plt.figure(figsize=(10, 6))
                plt.plot(x_data, y_data, 'o-', color=color)
                plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_').replace('/', '_')}.png"), dpi=300); plt.close()
        
        csv_data = {param_name: param_values, 'Centroid_Error_Px': centroid_errors, 'Centroid_Error_Std': centroid_stds, 
                    'Vector_Error_Arcsec': vector_errors, 'Vector_Error_Std': vector_stds, 'Success_Rate': success_rates}
        if param_name == 'focal_length':
            csv_data['FOV_Degrees'] = fovs; csv_data['Pixel_Resolution_Arcsec'] = pixel_resolutions
        pd.DataFrame(csv_data).to_csv(os.path.join(output_dir, f"{param_name}_analysis_results.csv"), index=False)
        logger.info(f"Saved {param_name} analysis results to {output_dir}")

    def run_temperature_analysis(self, temperatures, magnitude=None, photon_count=None,
                            num_trials=30, psf_data=None, output_dir=None):
        logger.info(f"Running temperature analysis with {len(temperatures)} temperatures")
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        if psf_data is None: logger.error("PSF data is required"); return None
        
        field_angles = sorted(psf_data.keys())
        on_axis_idx = np.argmin(np.abs(np.array(field_angles)))
        test_psf = psf_data[field_angles[on_axis_idx]]
        logger.info(f"Using PSF with field angle {field_angles[on_axis_idx]}° for analysis")
        original_temp = self.scene.temp
        
        temp_analysis_results = []
        for temp in temperatures:
            logger.info(f"Testing temperature = {temp}°C")
            temp_dir = os.path.join(output_dir, f"temp_{temp:.1f}C") if output_dir else None
            if temp_dir: os.makedirs(temp_dir, exist_ok=True)
            self.scene.temp = temp
            dark_current = self.camera.fpa.dark_current_ref * 2**((temp - self.camera.fpa.dark_current_ref_temp) / self.camera.fpa.dark_current_coefficient) * (self.scene.int_time / 1000)
            mc_results = self.run_monte_carlo_simulation(test_psf, magnitude=magnitude, photon_count=photon_count, num_trials=num_trials)
            
            result_item = {
                'temperature': temp, 'dark_current': dark_current,
                'centroid_error_px': mc_results['mean_centroid_error_px'], 
                'centroid_error_std': mc_results['std_centroid_error_px'], 
                'vector_error_arcsec': mc_results['mean_vector_error_arcsec'], 
                'vector_error_std': mc_results['std_vector_error_arcsec'], 
                'success_rate': mc_results['success_rate'], 
                'mc_results': mc_results
            }
            temp_analysis_results.append(result_item)
            if temp_dir: self.visualize_single_parameter_results(mc_results, temp_dir)
            
        self.scene.temp = original_temp
        if output_dir and temp_analysis_results:
            self.visualize_temperature_analysis_results(temp_analysis_results, output_dir)
        return {'temperatures': temperatures, 'results': temp_analysis_results}

    def visualize_temperature_analysis_results(self, temp_analysis_results, output_dir):
        import matplotlib.pyplot as plt
        temperatures = [r['temperature'] for r in temp_analysis_results]
        dark_currents = [r['dark_current'] for r in temp_analysis_results]
        centroid_errors = [r['centroid_error_px'] for r in temp_analysis_results]
        centroid_stds = [r['centroid_error_std'] for r in temp_analysis_results] 
        vector_errors = [r['vector_error_arcsec'] for r in temp_analysis_results]
        vector_stds = [r['vector_error_std'] for r in temp_analysis_results] 
        success_rates = [r['success_rate'] for r in temp_analysis_results]

        plot_configs = [
            ('Centroid Error vs Temperature', 'Centroid Error (pixels)', centroid_errors, centroid_stds, 'blue', "centroid_error_vs_temperature.png"),
            ('Bearing Vector Error vs Temperature', 'Angular Error (arcsec)', vector_errors, vector_stds, 'green', "vector_error_vs_temperature.png"),
            ('Success Rate vs Temperature', 'Success Rate', success_rates, None, 'purple', "success_rate_vs_temperature.png")
        ]

        for title, ylabel, data, stds, color, filename in plot_configs:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Temperature (°C)'); ax1.set_ylabel(ylabel, color=color)
            if stds: ax1.errorbar(temperatures, data, yerr=stds, fmt='o-', color=color, capsize=5)
            else: ax1.plot(temperatures, data, 'o-', color=color)
            if title == 'Success Rate vs Temperature': ax1.set_ylim(0, 1.05)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Dark Current (e-)', color='red'); ax2.plot(temperatures, dark_currents, 's--', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.title(title); plt.grid(True, alpha=0.3); fig.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300); plt.close()

        plt.figure(figsize=(10, 6))
        plt.semilogy(temperatures, dark_currents, 'o-', color='red')
        plt.title('Dark Current vs Temperature'); plt.xlabel('Temperature (°C)'); plt.ylabel('Dark Current (e-)')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dark_current_vs_temperature.png"), dpi=300); plt.close()
        
        df = pd.DataFrame({'Temperature_C': temperatures, 'Dark_Current_e': dark_currents, 
                           'Centroid_Error_Px': centroid_errors, 'Centroid_Error_Std': centroid_stds, 
                           'Vector_Error_Arcsec': vector_errors, 'Vector_Error_Std': vector_stds, 
                           'Success_Rate': success_rates})
        df.to_csv(os.path.join(output_dir, "temperature_analysis_results.csv"), index=False)
        logger.info(f"Saved temperature analysis results to {output_dir}")

    def run_magnitude_sweep(self, magnitudes=None, psf_data=None, 
                      num_trials=30, output_dir=None):
        if magnitudes is None: magnitudes = np.linspace(0.0, 6.0, 13)
        logger.info(f"Running magnitude sweep with {len(magnitudes)} magnitudes")
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        if psf_data is None: logger.error("PSF data is required"); return None
        
        field_angles = sorted(psf_data.keys())
        on_axis_idx = np.argmin(np.abs(np.array(field_angles)))
        test_psf = psf_data[field_angles[on_axis_idx]]
        logger.info(f"Using PSF with field angle {field_angles[on_axis_idx]}° for analysis")
        
        mag_sweep_results = []
        for mag in magnitudes:
            logger.info(f"Testing magnitude = {mag:.1f}")
            mag_dir = os.path.join(output_dir, f"mag_{mag:.1f}") if output_dir else None
            if mag_dir: os.makedirs(mag_dir, exist_ok=True)
            mc_results = self.run_monte_carlo_simulation(test_psf, magnitude=mag, num_trials=num_trials)
            photon_count_val = calculate_optical_signal(star(magnitude=mag, passband=self.camera.passband), self.camera, self.scene)
            
            result_item = {
                'magnitude': mag, 'photon_count': photon_count_val,
                'centroid_error_px': mc_results['mean_centroid_error_px'], 
                'centroid_error_std': mc_results['std_centroid_error_px'],
                'vector_error_arcsec': mc_results['mean_vector_error_arcsec'], 
                'vector_error_std': mc_results['std_vector_error_arcsec'],
                'success_rate': mc_results['success_rate'], 
                'mc_results': mc_results
            }
            mag_sweep_results.append(result_item)
            if mag_dir: self.visualize_single_parameter_results(mc_results, mag_dir)
            
        if output_dir and mag_sweep_results:
            self.visualize_magnitude_sweep_results(mag_sweep_results, output_dir)
        return {'magnitudes': magnitudes, 'results': mag_sweep_results}

    def visualize_magnitude_sweep_results(self, mag_sweep_results, output_dir):
        import matplotlib.pyplot as plt
        magnitudes = [r['magnitude'] for r in mag_sweep_results]
        photon_counts = [r['photon_count'] for r in mag_sweep_results]
        centroid_errors = [r['centroid_error_px'] for r in mag_sweep_results]
        centroid_stds = [r['centroid_error_std'] for r in mag_sweep_results] 
        vector_errors = [r['vector_error_arcsec'] for r in mag_sweep_results]
        vector_stds = [r['vector_error_std'] for r in mag_sweep_results] 
        success_rates = [r['success_rate'] for r in mag_sweep_results]

        plot_configs = [
            ('Centroid Error vs Magnitude', 'Star Magnitude', 'Centroid Error (pixels)', centroid_errors, centroid_stds, 'blue', "centroid_error_vs_magnitude.png"),
            ('Bearing Vector Error vs Magnitude', 'Star Magnitude', 'Angular Error (arcsec)', vector_errors, vector_stds, 'green', "vector_error_vs_magnitude.png"),
            ('Success Rate vs Magnitude', 'Star Magnitude', 'Success Rate', success_rates, None, 'purple', "success_rate_vs_magnitude.png"),
            ('Photon Count vs Magnitude', 'Star Magnitude', 'Photon Count', photon_counts, None, 'red', "photon_count_vs_magnitude.png", True) # is_semilogy
        ]

        for title, xlabel, ylabel, data, stds, color, filename, *semilogy_arg in plot_configs:
            is_semilogy = semilogy_arg[0] if semilogy_arg else False
            plt.figure(figsize=(10, 6))
            if stds: plt.errorbar(magnitudes, data, yerr=stds, fmt='o-', color=color, capsize=5)
            elif is_semilogy: plt.semilogy(magnitudes, data, 'o-', color=color)
            else: plt.plot(magnitudes, data, 'o-', color=color)
            plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
            if title == 'Success Rate vs Magnitude': plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300); plt.close()

        plt.figure(figsize=(10, 6))
        photon_counts_arr = np.array(photon_counts)
        valid_photon_mask = photon_counts_arr > 0
        if np.any(valid_photon_mask):
            inv_sqrt_photons = 1 / np.sqrt(photon_counts_arr[valid_photon_mask])
            vector_errors_filtered = np.array(vector_errors)[valid_photon_mask]
            plt.plot(inv_sqrt_photons, vector_errors_filtered, 'o-', color='green')
            if len(inv_sqrt_photons) > 1:
                from scipy import stats
                finite_mask = np.isfinite(inv_sqrt_photons) & np.isfinite(vector_errors_filtered)
                if np.sum(finite_mask) > 1:
                    slope, intercept, r_value, _, _ = stats.linregress(inv_sqrt_photons[finite_mask], vector_errors_filtered[finite_mask])
                    x_fit = np.linspace(min(inv_sqrt_photons[finite_mask]), max(inv_sqrt_photons[finite_mask]), 100)
                    plt.plot(x_fit, slope * x_fit + intercept, '--', color='red', label=f'Fit: Error = {slope:.2f} * 1/√photons + {intercept:.2f}')
                    plt.text(0.05, 0.95, f'R² = {r_value**2:.4f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
                else: logger.warning("Not enough finite data for linreg in mag sweep viz.")
            plt.legend()
        plt.title('Vector Error vs 1/√(Photon Count)'); plt.xlabel('1/√(Photon Count)'); plt.ylabel('Angular Error (arcsec)')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_vs_inv_sqrt_photons.png"), dpi=300); plt.close()
        
        df = pd.DataFrame({'Magnitude': magnitudes, 'Photon_Count': photon_counts, 'Centroid_Error_Px': centroid_errors, 
                           'Centroid_Error_Std': centroid_stds, 'Vector_Error_Arcsec': vector_errors, 
                           'Vector_Error_Std': vector_stds, 'Success_Rate': success_rates})
        df.to_csv(os.path.join(output_dir, "magnitude_sweep_results.csv"), index=False)
        logger.info(f"Saved magnitude sweep results to {output_dir}")

    def analyze_psf_characteristics(self, psf_data_dict, output_dir=None):
        """
        Analyze characteristics of PSF files across field angles.
        
        Args:
            psf_data_dict: Dictionary mapping field angles to PSF data (each item contains intensity_data, metadata, file_path)
            output_dir: Directory to save analysis results
            
        Returns:
            dict: Analysis results for each PSF
        """
        results = {}
        
        for field_angle in sorted(psf_data_dict.keys()):
            psf_item = psf_data_dict[field_angle]
            intensity_data = psf_item['intensity_data']
            
            # Calculate various PSF metrics
            peak_intensity = np.max(intensity_data)
            total_intensity = np.sum(intensity_data)
            
            # Calculate centroid using the (potentially) new method
            centroid_x, centroid_y = self.calculate_true_psf_centroid(psf_item)
            
            # Calculate second moments (spread)
            y_indices, x_indices = np.indices(intensity_data.shape)
            x_centered = x_indices - centroid_x
            y_centered = y_indices - centroid_y
            
            # Intensity-weighted second moments
            sigma_xx = np.sum(x_centered**2 * intensity_data) / total_intensity
            sigma_yy = np.sum(y_centered**2 * intensity_data) / total_intensity
            sigma_xy = np.sum(x_centered * y_centered * intensity_data) / total_intensity
            
            # RMS radius
            rms_radius = np.sqrt(sigma_xx + sigma_yy)
            
            # Eccentricity from moments
            det = sigma_xx * sigma_yy - sigma_xy**2
            trace = sigma_xx + sigma_yy
            if det > 0 and trace > 0:
                lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det))
                lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det))
                eccentricity = np.sqrt(1 - lambda2/lambda1) if lambda1 > 0 else 0
            else:
                eccentricity = 0
            
            # Store results
            results[field_angle] = {
                'peak_intensity': peak_intensity,
                'total_intensity': total_intensity,
                'centroid': (centroid_x, centroid_y),
                'rms_radius': rms_radius,
                'sigma_xx': sigma_xx,
                'sigma_yy': sigma_yy,
                'eccentricity': eccentricity,
                'shape': intensity_data.shape
            }
        
        # Visualize if output directory provided
        if output_dir:
            import matplotlib.pyplot as plt
            os.makedirs(output_dir, exist_ok=True)
            
            field_angles = sorted(results.keys())
            
            # Plot RMS radius vs field angle
            plt.figure(figsize=(10, 6))
            rms_radii = [results[fa]['rms_radius'] for fa in field_angles]
            plt.plot(field_angles, rms_radii, 'o-')
            plt.xlabel('Field Angle (degrees)')
            plt.ylabel('PSF RMS Radius (pixels)')
            plt.title('PSF Size vs Field Angle')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'psf_rms_radius_vs_angle.png'), dpi=300)
            plt.close()
            
            # Plot eccentricity vs field angle
            plt.figure(figsize=(10, 6))
            eccentricities = [results[fa]['eccentricity'] for fa in field_angles]
            plt.plot(field_angles, eccentricities, 'o-')
            plt.xlabel('Field Angle (degrees)')
            plt.ylabel('PSF Eccentricity')
            plt.title('PSF Eccentricity vs Field Angle')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'psf_eccentricity_vs_angle.png'), dpi=300)
            plt.close()
            
            # Save detailed results
            import pandas as pd
            df = pd.DataFrame(results).T
            df.to_csv(os.path.join(output_dir, 'psf_characteristics.csv'))
            
        return results

    def project_psf_to_fpa_grid(self, psf_data, target_pixel_pitch_um=5.5, create_full_fpa=False, fpa_size=(2048, 2048), apply_coordinate_shift=False):
        """
        Project PSF from simulation grid to FPA pixel grid by integrating intensity over pixel blocks.
        Uses skimage.measure.block_reduce for robust handling of differing dimensions.
        
        Optionally applies a 2.75μm coordinate shift (half CMV4000 pixel pitch) to center the PSF on pixel
        centers rather than pixel corners for worst-case centroiding scenario testing.
        
        Args:
            psf_data: PSF data dict containing intensity_data and metadata
            target_pixel_pitch_um: Target FPA pixel pitch in microns (default 5.5 for CMV4000)
            create_full_fpa: If True, create full FPA grid (2048×2048) with PSF placed randomly (default False)
            fpa_size: Size of full FPA grid (height, width) for CMV4000 (default (2048, 2048))
            apply_coordinate_shift: If True, apply 2.75μm shift to center PSF on pixel center (default False)
            
        Returns:
            dict: FPA-projected PSF data with updated intensity_data, metadata, and scaling info
        """
        psf_intensity = psf_data['intensity_data']
        psf_metadata = psf_data['metadata']
        
        # --- BEGIN DEBUG PRINT ---
        if self.debug and psf_metadata:
            ds_val_proj = psf_metadata.get('data_spacing')
            ds_type_proj = type(ds_val_proj)
            logger.info(f"DEBUG project_psf_to_fpa_grid: Received metadata['data_spacing'] = {ds_val_proj}, type: {ds_type_proj} for file {psf_data.get('file_path')}")
        elif self.debug:
            logger.info(f"DEBUG project_psf_to_fpa_grid: psf_metadata missing/empty for file {psf_data.get('file_path')}")
        # --- END DEBUG PRINT ---
        
        # Get PSF simulation pixel spacing
        psf_pixel_spacing_um = self._get_psf_data_spacing_microns(psf_metadata)
        
        if psf_pixel_spacing_um <= 0:
            logger.error(f"Invalid PSF pixel spacing: {psf_pixel_spacing_um} µm")
            return psf_data
        
        # Calculate scaling factor
        scale_factor = target_pixel_pitch_um / psf_pixel_spacing_um
        scale_factor_int = int(np.round(scale_factor))
        
        # Optional coordinate shift: 2.75μm (half pixel pitch) to center PSF on pixel centers
        if apply_coordinate_shift:
            coordinate_shift_um = target_pixel_pitch_um / 2.0  # 2.75μm for CMV4000
            coordinate_shift_pixels = coordinate_shift_um / psf_pixel_spacing_um
        else:
            coordinate_shift_um = 0.0
            coordinate_shift_pixels = 0.0
        
        logger.info(f"PSF pixel spacing: {psf_pixel_spacing_um} µm")
        logger.info(f"Target FPA pixel pitch: {target_pixel_pitch_um} µm")
        logger.info(f"Scale factor: {scale_factor:.2f} → {scale_factor_int} (rounded)")
        if apply_coordinate_shift:
            logger.info(f"Applying coordinate shift: {coordinate_shift_um:.2f} µm ({coordinate_shift_pixels:.2f} PSF pixels) - worst-case scenario")
        else:
            logger.info(f"No coordinate shift applied - best-case scenario (PSF center on pixel corner)")
        
        # Original PSF dimensions
        orig_height, orig_width = psf_intensity.shape
        
        # Calculate new FPA grid dimensions - this will be determined by block_reduce
        # fpa_height = orig_height // scale_factor_int # Old method
        # fpa_width = orig_width // scale_factor_int  # Old method
        
        logger.info(f"Original PSF grid: {orig_height}x{orig_width} pixels")
        # logger.info(f"FPA grid: {fpa_height}x{fpa_width} pixels") # Will be logged after block_reduce
        
        # Calculate physical areas - original PSF physical area
        orig_physical_height_um = orig_height * psf_pixel_spacing_um
        orig_physical_width_um = orig_width * psf_pixel_spacing_um
        logger.info(f"Original PSF physical size: {orig_physical_height_um:.2f} x {orig_physical_width_um:.2f} µm")

        if scale_factor_int <= 0:
            logger.error(f"Calculated integer scale factor ({scale_factor_int}) is zero or negative. Cannot perform block reduction. PSF spacing: {psf_pixel_spacing_um}, Target FPA spacing: {target_pixel_pitch_um}")
            # Return a minimal 1x1 FPA PSF to avoid crashing downstream, but clearly indicate error
            fpa_intensity = np.array([[np.sum(psf_intensity)]]) # Put all energy in one pixel
            fpa_metadata = psf_metadata.copy() if psf_metadata else {}
            fpa_metadata.update({
                'error_in_projection': True,
                'data_spacing_um': target_pixel_pitch_um,
                'image_grid_dim': (1,1),
                'fpa_projected': True
            })
            return {
                'intensity_data': fpa_intensity,
                'full_fpa_intensity': None,
                'metadata': fpa_metadata,
                'file_path': psf_data['file_path'],
                'scaling_info': {
                    'error': 'Invalid scale_factor_int',
                    'original_shape': (orig_height, orig_width),
                    'fpa_shape': (1,1),
                    'original_pixel_spacing_um': psf_pixel_spacing_um,
                    'fpa_pixel_spacing_um': target_pixel_pitch_um,
                    'scale_factor_int_used': scale_factor_int,
                    'intensity_conservation_ratio': 1.0 # Since all energy is summed
                }
            }

        # Check if PSF is too small for meaningful block reduction (Gen 2 case)
        min_required_pixels = scale_factor_int * 3  # Need at least 3x3 output
        use_interpolation = orig_height < min_required_pixels or orig_width < min_required_pixels
        
        # Force interpolation for Gen 4, 5, 6 based on small PSF grid size
        if psf_data.get('file_path') and any(gen in psf_data['file_path'] for gen in ["Gen_4", "Gen_5", "Gen_6"]):
            if orig_height < 100 and orig_width < 100: # Heuristic for these specific gens
                logger.info("Forcing interpolation for Gen 4/5/6 due to small grid size.")
                use_interpolation = True

        if use_interpolation:
            logger.warning(f"PSF grid ({orig_height}×{orig_width}) too small for scale factor {scale_factor_int} or is Gen 4/5/6.")
            logger.info(f"Using interpolation-based approach for fine-resolution PSF")
            
            # For fine-resolution PSFs (like Gen 2), use interpolation approach with coordinate shift
            # Calculate target FPA size based on physical area
            psf_physical_height_um = orig_height * psf_pixel_spacing_um
            psf_physical_width_um = orig_width * psf_pixel_spacing_um
            
            # Target FPA size in pixels
            fpa_height_exact = psf_physical_height_um / target_pixel_pitch_um
            fpa_width_exact = psf_physical_width_um / target_pixel_pitch_um
            
            fpa_height = max(3, int(np.round(fpa_height_exact)))  # Ensure minimum size
            fpa_width = max(3, int(np.round(fpa_width_exact)))
            
            # Ensure odd dimensions for proper centering
            if fpa_height % 2 == 0: fpa_height += 1
            if fpa_width % 2 == 0: fpa_width += 1
            
            logger.info(f"Interpolating to FPA grid: {fpa_height}×{fpa_width} pixels")
            
            # Apply coordinate shift by using scipy.ndimage.shift followed by interpolation
            from scipy.ndimage import shift, zoom
            
            # Apply coordinate shift if requested
            if apply_coordinate_shift:
                # Shift by half an FPA pixel in PSF coordinates
                shifted_psf = shift(psf_intensity, (coordinate_shift_pixels, coordinate_shift_pixels), 
                                  order=3, mode='constant', cval=0.0)
                
                # Ensure no negative values after shift
                shifted_psf = np.maximum(shifted_psf, 0.0)
                
                # Preserve total intensity
                orig_sum = np.sum(psf_intensity)
                shifted_sum = np.sum(shifted_psf)
                if shifted_sum > 0:
                    shifted_psf = shifted_psf * (orig_sum / shifted_sum)
                    
                logger.info(f"Applied coordinate shift: intensity conservation = {np.sum(shifted_psf)/orig_sum:.6f}")
            else:
                shifted_psf = psf_intensity
            
            # Then interpolate the (possibly shifted) PSF to FPA grid
            zoom_factor_y = fpa_height / orig_height
            zoom_factor_x = fpa_width / orig_width
            
            fpa_intensity = zoom(shifted_psf, (zoom_factor_y, zoom_factor_x), order=3)  # Cubic interpolation
            
            # Normalize to preserve total intensity
            orig_total = np.sum(psf_intensity)
            fpa_total = np.sum(fpa_intensity)
            if fpa_total > 0:
                fpa_intensity = fpa_intensity * (orig_total / fpa_total)
            
            intensity_ratio = np.sum(fpa_intensity) / orig_total if orig_total > 0 else 0
            
            logger.info(f"Interpolated PSF physical size: {fpa_height * target_pixel_pitch_um:.2f} x {fpa_width * target_pixel_pitch_um:.2f} µm")
            logger.info(f"Total intensity: Original={orig_total:.2e}, FPA={np.sum(fpa_intensity):.2e}, Ratio={intensity_ratio:.4f}")
            
        else:
            # Standard block reduction approach for larger PSFs (Gen 1 case) with coordinate shift
            logger.info(f"Using block reduction approach for standard PSF")
            
            # Apply coordinate shift by shifting the data before block reduction
            # For block reduction, we need to pad the array to handle the shift properly
            from scipy.ndimage import shift
            
            # Apply coordinate shift if requested
            if apply_coordinate_shift:
                shifted_psf = shift(psf_intensity, (coordinate_shift_pixels, coordinate_shift_pixels), 
                                  order=3, mode='constant', cval=0.0)
                
                # Ensure no negative values after shift
                shifted_psf = np.maximum(shifted_psf, 0.0)
                
                # Preserve total intensity
                orig_sum = np.sum(psf_intensity)
                shifted_sum = np.sum(shifted_psf)
                if shifted_sum > 0:
                    shifted_psf = shifted_psf * (orig_sum / shifted_sum)
                    
                logger.info(f"Applied coordinate shift: intensity conservation = {np.sum(shifted_psf)/orig_sum:.6f}")
            else:
                shifted_psf = psf_intensity
            
            # Create FPA-projected intensity grid by summing over blocks using skimage.block_reduce
            # The block_size for block_reduce is how many original pixels form one new FPA pixel.
            block_dims = (scale_factor_int, scale_factor_int)
            fpa_intensity = block_reduce(shifted_psf, block_size=block_dims, func=np.sum, cval=0)
            
            # Calculate intensity conservation
            orig_total = np.sum(psf_intensity)
            fpa_total = np.sum(fpa_intensity)
            intensity_ratio = fpa_total / orig_total if orig_total > 0 else 0
        
        fpa_height, fpa_width = fpa_intensity.shape
        logger.info(f"FPA projected grid: {fpa_height}x{fpa_width} pixels")

        # Physical area of the effectively sampled region on the FPA
        fpa_physical_height_um = fpa_height * target_pixel_pitch_um
        fpa_physical_width_um = fpa_width * target_pixel_pitch_um
        logger.info(f"FPA projected physical size: {fpa_physical_height_um:.2f} x {fpa_physical_width_um:.2f} µm")
        
        # Create full FPA grid if requested
        full_fpa_intensity = None
        fpa_position = None
        if create_full_fpa:
            full_fpa_height, full_fpa_width = fpa_size
            
            # Check if FPA PSF fits in full detector
            if fpa_height > full_fpa_height or fpa_width > full_fpa_width:
                logger.warning(f"FPA PSF ({fpa_height}×{fpa_width}) too large for full detector ({full_fpa_height}×{full_fpa_width})")
                logger.warning("Skipping full FPA generation")
            else:
                # Create full FPA grid (all zeros)
                full_fpa_intensity = np.zeros((full_fpa_height, full_fpa_width))
                
                # Choose random position that keeps PSF fully within bounds
                max_row_start = full_fpa_height - fpa_height
                max_col_start = full_fpa_width - fpa_width
                
                row_start = np.random.randint(0, max_row_start + 1) if max_row_start > 0 else 0
                col_start = np.random.randint(0, max_col_start + 1) if max_col_start > 0 else 0
                
                # Place the FPA PSF at the random position
                full_fpa_intensity[row_start:row_start+fpa_height, col_start:col_start+fpa_width] = fpa_intensity
                
                fpa_position = (row_start, col_start)
                
                logger.info(f"Created full FPA grid: {full_fpa_height}×{full_fpa_width} pixels")
                logger.info(f"PSF placed at position ({row_start}, {col_start}) to ({row_start+fpa_height-1}, {col_start+fpa_width-1})")
                
                # Calculate physical coordinates
                row_start_mm = (row_start - full_fpa_height/2) * target_pixel_pitch_um / 1000  # Convert to mm
                col_start_mm = (col_start - full_fpa_width/2) * target_pixel_pitch_um / 1000  # Convert to mm
                logger.info(f"PSF physical position: ({row_start_mm:.2f}, {col_start_mm:.2f}) mm from detector center")
        
        # Create new metadata
        fpa_metadata = psf_metadata.copy() if psf_metadata else {}
        fpa_metadata.update({
            'data_spacing_um': target_pixel_pitch_um,
            'data_spacing': f"{target_pixel_pitch_um:.3f} um",
            'image_grid_dim': (fpa_width, fpa_height), # Added: (width, height) of the FPA-projected PSF array
            'image_grid_size': f"{fpa_height} by {fpa_width}", # Existing: string H x W
            'original_data_spacing_um': psf_pixel_spacing_um,
            'original_image_grid_dim': (orig_width, orig_height), # Storing original W, H
            'scale_factor_used': scale_factor_int, # The integer factor used for block_reduce blocks
            'fpa_projected': True,
            'coordinate_shift_applied': apply_coordinate_shift,
            'coordinate_shift_um': coordinate_shift_um,
            'coordinate_shift_pixels': coordinate_shift_pixels,
            'fpa_grid_size': f"{fpa_height} by {fpa_width}", # Redundant with image_grid_size for the small PSF, but consistent
            'full_fpa_created': create_full_fpa,
            'full_fpa_size': f"{fpa_size[0]} by {fpa_size[1]}" if create_full_fpa else None,
            'fpa_position': fpa_position,
            'fpa_physical_size_um_str': f"{fpa_physical_height_um:.3f} x {fpa_physical_width_um:.3f}",
        })
        
        # Validate FPA intensity data before returning
        if np.any(np.isnan(fpa_intensity)) or np.any(np.isinf(fpa_intensity)):
            logger.error("FPA intensity contains NaN or Inf values after projection")
            logger.error(f"Original PSF stats: min={np.min(psf_intensity)}, max={np.max(psf_intensity)}, sum={np.sum(psf_intensity)}")
            logger.error(f"FPA PSF stats: min={np.min(fpa_intensity)}, max={np.max(fpa_intensity)}, sum={np.sum(fpa_intensity)}")
            # Set any invalid values to zero
            fpa_intensity = np.nan_to_num(fpa_intensity, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(fpa_intensity < 0):
            logger.warning("FPA intensity contains negative values - clipping to zero")
            logger.warning(f"Min value: {np.min(fpa_intensity)}")
            fpa_intensity = np.maximum(fpa_intensity, 0.0)
        
        # Create FPA-projected PSF data
        fpa_psf_data = {
            'intensity_data': fpa_intensity,  # This is the 11×11 grid
            'full_fpa_intensity': full_fpa_intensity,  # This is the 2048×2048 grid (or None)
            'metadata': fpa_metadata,
            'file_path': psf_data['file_path'],
            'scaling_info': {
                'original_shape': (orig_height, orig_width),
                'fpa_shape': (fpa_height, fpa_width),
                'full_fpa_shape': fpa_size if create_full_fpa else None,
                'fpa_position': fpa_position,
                'original_pixel_spacing_um': psf_pixel_spacing_um,
                'fpa_pixel_spacing_um': target_pixel_pitch_um,
                'scale_factor_int_used': scale_factor_int if not use_interpolation else 'interpolation',
                'intensity_conservation_ratio': intensity_ratio,
                'projection_method': 'interpolation' if use_interpolation else 'block_reduction',
                'coordinate_shift_applied': apply_coordinate_shift,
                'coordinate_shift_um': coordinate_shift_um,
                'coordinate_shift_pixels': coordinate_shift_pixels
            }
        }
        
        return fpa_psf_data

    def run_monte_carlo_simulation_fpa_projected(self, psf_data, magnitude=None, photon_count=None,
                                                  num_trials=100, threshold_sigma=3.0,
                                                  adaptive_block_size=8, target_pixel_pitch_um=5.5,
                                                  create_full_fpa=False, fpa_size=(2048, 2048),
                                                  return_debug_data=False, apply_coordinate_shift=False):
        """
        Run Monte Carlo simulation on FPA-projected PSF data (e.g., for CMV4000).
        
        Args:
            psf_data: Original PSF data (will be projected to FPA grid internally)
            magnitude: Star magnitude (only used if photon_count is None)
            photon_count: Direct photon count (overrides magnitude if provided)
            num_trials: Number of Monte Carlo trials
            threshold_sigma: Sigma multiplier for detection threshold
            adaptive_block_size: Block size for adaptive thresholding (smaller for FPA grid)
            target_pixel_pitch_um: Target FPA pixel pitch in microns (default 5.5 for CMV4000)
            create_full_fpa: If True, create full FPA grid (2048×2048) with PSF placed randomly
            fpa_size: Size of full FPA grid (height, width) for CMV4000
            return_debug_data: If True, return intermediate data for debugging
            apply_coordinate_shift: If True, apply 2.75μm shift to center PSF on pixel center
            
        Returns:
            dict: Monte Carlo simulation results with FPA projection information
        """
        # Project PSF to FPA grid
        logger.info("Projecting PSF to FPA grid...")
        fpa_psf_data = self.project_psf_to_fpa_grid(psf_data, target_pixel_pitch_um, create_full_fpa, fpa_size, apply_coordinate_shift)
        
        # Log FPA projection details
        scaling_info = fpa_psf_data['scaling_info']
        logger.info(f"FPA projection: {scaling_info['original_shape']} → {scaling_info['fpa_shape']} pixels")
        logger.info(f"Pixel scaling: {scaling_info['original_pixel_spacing_um']:.1f} → {scaling_info['fpa_pixel_spacing_um']:.1f} µm/px")
        logger.info(f"Scale factor: {scaling_info['scale_factor_int_used']}")
        logger.info(f"Intensity conservation: {scaling_info['intensity_conservation_ratio']:.4f}")
        
        if create_full_fpa and scaling_info['full_fpa_shape'] is not None:
            logger.info(f"Full FPA grid: {scaling_info['full_fpa_shape']} pixels")
            if scaling_info['fpa_position'] is not None:
                logger.info(f"PSF position on full FPA: {scaling_info['fpa_position']}")
        
        # Simulate star
        star_simulation = self.simulate_star(magnitude, photon_count)
        photon_count_actual = star_simulation['photon_count'] 
        
        # Calculate the true centroid from the FPA-projected PSF (using the 11×11 grid)
        true_centroid_fpa = self.calculate_true_psf_centroid(fpa_psf_data, use_zemax_offsets=False)
        
        # Project star using FPA-projected PSF (this uses the 11×11 grid for simulation)
        projection_results = self.project_star_with_psf(
            fpa_psf_data, photon_count_actual, num_simulations=num_trials 
        )
        
        sim_image_shape = None
        if projection_results['simulations'] and len(projection_results['simulations']) > 0:
            sim_image_shape = projection_results['simulations'][0].shape
        
        # Adjust detection parameters for FPA grid size and PSF characteristics
        fpa_height, fpa_width = scaling_info['fpa_shape']
        fpa_grid_area = fpa_height * fpa_width
        fpa_intensity = fpa_psf_data['intensity_data']
        
        # Analyze PSF characteristics to adjust thresholding
        psf_max = np.max(fpa_intensity)
        psf_mean = np.mean(fpa_intensity)
        psf_std = np.std(fpa_intensity)
        
        # Calculate multiple metrics to detect degraded PSFs
        # 1. PSF spread (percentage of pixels with significant signal)
        significant_pixels = np.sum(fpa_intensity > psf_mean + 0.5 * psf_std)
        psf_spread_ratio = significant_pixels / fpa_grid_area
        
        # 2. Peak-to-mean ratio (lower for degraded PSFs)
        peak_to_mean_ratio = psf_max / psf_mean if psf_mean > 0 else 0
        
        # 3. Signal concentration (fraction of total intensity in brightest pixel)
        total_intensity = np.sum(fpa_intensity)
        concentration_ratio = psf_max / total_intensity if total_intensity > 0 else 0
        
        # Detect degraded PSFs using multiple criteria
        is_very_diffuse = (psf_spread_ratio > 0.5) or (peak_to_mean_ratio < 3.0) or (concentration_ratio < 0.1)
        is_moderately_diffuse = (psf_spread_ratio > 0.3) or (peak_to_mean_ratio < 5.0) or (concentration_ratio < 0.15)
        
        # Adjust threshold based on PSF quality - degraded PSFs need lower thresholds
        if is_very_diffuse:  # Very degraded PSF
            adjusted_threshold_sigma = max(2.0, threshold_sigma * 0.4)  # Much lower threshold
            logger.info(f"Very diffuse PSF detected (spread={psf_spread_ratio:.2f}, peak/mean={peak_to_mean_ratio:.1f}, concentration={concentration_ratio:.3f}), reducing threshold to {adjusted_threshold_sigma:.1f}σ")
        elif is_moderately_diffuse:  # Moderately degraded PSF
            adjusted_threshold_sigma = max(2.5, threshold_sigma * 0.6)  # Lower threshold
            logger.info(f"Diffuse PSF detected (spread={psf_spread_ratio:.2f}, peak/mean={peak_to_mean_ratio:.1f}, concentration={concentration_ratio:.3f}), reducing threshold to {adjusted_threshold_sigma:.1f}σ")
        else:  # Good quality PSF
            adjusted_threshold_sigma = threshold_sigma
        
        # Scale detection parameters based on grid size
        if fpa_grid_area < 200:  # Small grid (like 11x11 = 121 pixels)
            min_pixels_fpa = 1  # Allow very small regions
            max_pixels_fpa = max(20, fpa_grid_area // 2)  # Allow larger regions
            
            # For small grids, a small, odd block size is better.
            # A block size of 3 is usually effective and won't average over the whole image.
            block_size_fpa = 3
            
            # Ensure block size is not larger than the image itself
            if block_size_fpa >= min(fpa_height, fpa_width):
                block_size_fpa = min(fpa_height, fpa_width)
                if block_size_fpa % 2 == 0 and block_size_fpa > 1: # Make it odd if possible
                    block_size_fpa -= 1
                block_size_fpa = max(1, block_size_fpa) # Ensure it's at least 1

        else:  # Larger grids
            min_pixels_fpa = 3
            max_pixels_fpa = 50
            block_size_fpa = adaptive_block_size
        
        logger.info(f"FPA detection parameters: min_pixels={min_pixels_fpa}, max_pixels={max_pixels_fpa}, block_size={block_size_fpa}, threshold={adjusted_threshold_sigma:.1f}σ")
        
        # Use adjusted parameters for FPA detection
        detection_call_args = {
            'images': projection_results['simulations'],
            'k_sigma': adjusted_threshold_sigma,
            'min_pixels': min_pixels_fpa,
            'max_pixels': max_pixels_fpa,
            'block_size': block_size_fpa,
            'true_center': true_centroid_fpa,
            'return_debug_data': return_debug_data
        }
        
        detection_result = self.detect_stars_and_calculate_centroids(**detection_call_args)
        
        if return_debug_data:
            centroid_results, debug_info = detection_result
        else:
            centroid_results = detection_result
            debug_info = None
        
        # Initialize bearing_results
        bearing_results = { 
            'bearing_vectors': [], 'vector_errors_arcsec': [],
            'mean_vector_error_arcsec': float('nan'), 'std_vector_error_arcsec': float('nan')
        }

        if centroid_results['successful_detections'] > 0:
            if sim_image_shape is None:
                logger.error("sim_image_shape is None but detections were successful. Cannot calculate bearing vectors for FPA projected PSF.")
            else:
                # Calculate the true bearing vector from the FPA-projected PSF centroid position
                true_bearing_vector_results = self.calculate_bearing_vectors(
                    [true_centroid_fpa],  # The true centroid position in FPA coordinates
                    image_shape=sim_image_shape,
                    sensor_pixel_pitch_um=target_pixel_pitch_um, # Use FPA's pixel pitch
                    true_bearing_vector=None  # Don't compare to anything yet
                )
                
                if true_bearing_vector_results['bearing_vectors']:
                    true_bearing_vector = true_bearing_vector_results['bearing_vectors'][0]
                else:
                    logger.warning("Failed to calculate true_bearing_vector for FPA projected PSF. Defaulting to boresight.")
                    true_bearing_vector = np.array([0.0, 0.0, 1.0]) # Fallback to boresight
                
                # Calculate bearing vectors for detected centroids and compare to true
                bearing_results = self.calculate_bearing_vectors(
                    centroid_results['centroids'], 
                    image_shape=sim_image_shape,
                    sensor_pixel_pitch_um=target_pixel_pitch_um, # Use FPA's pixel pitch
                    true_bearing_vector=true_bearing_vector
                )
        
        # Calculate centroiding errors in metric space (microns)
        centroid_errors_um = []
        if centroid_results['centroid_errors']:
            # Convert pixel errors to micron errors
            centroid_errors_um = [error * target_pixel_pitch_um for error in centroid_results['centroid_errors']]
        
        # Print immediate status
        if centroid_results['successful_detections'] > 0:
            logger.info(f"    FPA Successful detections: {centroid_results['successful_detections']}/{num_trials}")
            logger.info(f"    FPA Mean centroid error: {centroid_results['mean_error']:.3f} FPA pixels ({centroid_results['mean_error'] * target_pixel_pitch_um:.2f} µm)")
            if 'mean_vector_error_arcsec' in bearing_results and not np.isnan(bearing_results['mean_vector_error_arcsec']):
                logger.info(f"    FPA Mean vector error: {bearing_results['mean_vector_error_arcsec']:.2f} arcsec")
        else:
            logger.warning(f"    No successful FPA detections for this field angle")

        # Combine all results
        results = {
            'star_simulation': star_simulation,
            'fpa_psf_data': fpa_psf_data,
            'projection_results': projection_results,
            'centroid_results': centroid_results,
            'bearing_results': bearing_results,
            'successful_trials': centroid_results['successful_detections'],
            'num_trials': num_trials,
            'success_rate': centroid_results['success_rate'],
            'mean_centroid_error_px': centroid_results['mean_error'],
            'std_centroid_error_px': centroid_results['std_error'],
            'mean_centroid_error_um': np.mean(centroid_errors_um) if centroid_errors_um else float('nan'),
            'std_centroid_error_um': np.std(centroid_errors_um) if centroid_errors_um else float('nan'),
            'mean_vector_error_arcsec': bearing_results['mean_vector_error_arcsec'],
            'std_vector_error_arcsec': bearing_results['std_vector_error_arcsec'],
            'fpa_pixel_pitch_um': target_pixel_pitch_um,
            'scaling_info': scaling_info,
            'detection_params': {
                'min_pixels': min_pixels_fpa,
                'max_pixels': max_pixels_fpa,
                'block_size': block_size_fpa
            },
            'full_fpa_created': create_full_fpa,
            'debug_data': debug_info
        }
        
        # Calculate confidence intervals (95%)
        if centroid_results['successful_detections'] > 2: 
            if not np.isnan(results['std_centroid_error_px']):
                 results['centroid_error_95ci_px'] = 1.96 * results['std_centroid_error_px'] / np.sqrt(centroid_results['successful_detections'])
            else:
                results['centroid_error_95ci_px'] = float('nan')
                
            if not np.isnan(results['std_centroid_error_um']):
                 results['centroid_error_95ci_um'] = 1.96 * results['std_centroid_error_um'] / np.sqrt(centroid_results['successful_detections'])
            else:
                results['centroid_error_95ci_um'] = float('nan')

            if not np.isnan(results['std_vector_error_arcsec']):
                results['vector_error_95ci'] = 1.96 * results['std_vector_error_arcsec'] / np.sqrt(centroid_results['successful_detections'])
            else:
                results['vector_error_95ci'] = float('nan')
        else:
            results['centroid_error_95ci_px'] = float('nan')
            results['centroid_error_95ci_um'] = float('nan')
            results['vector_error_95ci'] = float('nan')
        
        return results

def process_single_psf(pipeline, psf_file, magnitude=None, photon_count=None, 
                      num_trials=50, output_dir=None):
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    try:
        metadata, intensity_data = parse_psf_file(psf_file)
        current_psf_data = {'metadata': metadata, 'intensity_data': intensity_data, 'file_path': psf_file}
    except Exception as e: logger.error(f"Error loading PSF {psf_file}: {e}"); return None
    
    field_angle = metadata.get('field_angle', 0.0)
    logger.info(f"Processing PSF with field angle {field_angle}°")
    # Use the core pipeline method for simulation
    mc_results = pipeline.run_monte_carlo_simulation(current_psf_data, magnitude=magnitude, photon_count=photon_count, num_trials=num_trials)
    
    if output_dir and mc_results['successful_trials'] > 0:
        # Visualize results using a method from the pipeline or a dedicated plotting function
        pipeline.visualize_single_parameter_results(mc_results, output_dir) # Assuming this is suitable
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"PSF File: {os.path.basename(psf_file)}\nField Angle: {field_angle}°\n")
            if magnitude is not None: f.write(f"Star Magnitude: {magnitude}\n")
            if mc_results.get('star_simulation'): # Check if star_simulation exists
                f.write(f"Photon Count: {mc_results['star_simulation'].get('photon_count', 'N/A'):.1f}\n")
            else:
                f.write("Photon Count: N/A (star_simulation data missing)\n")
            f.write(f"Trials: {num_trials}, Successful: {mc_results['successful_trials']}, Rate: {mc_results['success_rate']:.2f}\n")
            f.write(f"Centroid Error: {mc_results['mean_centroid_error_px']:.3f} ± {mc_results['std_centroid_error_px']:.3f} px\n")
            # Also show micron error if available
            if 'mean_centroid_error_um' in mc_results and not np.isnan(mc_results['mean_centroid_error_um']):
                 f.write(f"Centroid Error (µm): {mc_results['mean_centroid_error_um']:.3f} ± {mc_results.get('std_centroid_error_um', float('nan распознавание')):.3f} µm\n")
            f.write(f"Vector Error: {mc_results['mean_vector_error_arcsec']:.2f} ± {mc_results['std_vector_error_arcsec']:.2f} arcsec\n")
        logger.info(f"Saved results to {output_dir}")
    elif mc_results['successful_trials'] == 0: logger.warning("No successful trials.")
    return mc_results

def batch_process_psf_files(pipeline, psf_directory, pattern="*_deg.txt", # Updated default pattern
                           magnitude=None, photon_count=None, num_trials=50, output_dir=None):
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    all_psf_data = pipeline.load_psf_data(psf_directory, pattern)
    if not all_psf_data: logger.error("No PSF files found"); return None
    
    batch_results = {'field_angles': [], 'centroid_errors': [], 'centroid_stds': [], 
                     'vector_errors': [], 'vector_stds': [], 'success_rates': [], 'individual_results': {}}
    
    for field_angle, current_psf_data in sorted(all_psf_data.items()):
        logger.info(f"Processing field angle {field_angle}°")
        angle_dir = os.path.join(output_dir, f"angle_{field_angle:.1f}") if output_dir else None
        if angle_dir: os.makedirs(angle_dir, exist_ok=True)
        mc_results = pipeline.run_monte_carlo_simulation(current_psf_data, magnitude=magnitude, photon_count=photon_count, num_trials=num_trials)
        
        for key, val_list_key in [('mean_centroid_error_px', 'centroid_errors'), ('std_centroid_error_px', 'centroid_stds'),
                                  ('mean_vector_error_arcsec', 'vector_errors'), ('std_vector_error_arcsec', 'vector_stds'),
                                  ('success_rate', 'success_rates')]:
            batch_results[val_list_key].append(mc_results[key])
        batch_results['field_angles'].append(field_angle)
        batch_results['individual_results'][field_angle] = mc_results
        if angle_dir: pipeline.visualize_single_parameter_results(mc_results, angle_dir)
            
    if output_dir and batch_results['field_angles']:
        visualize_batch_results(batch_results, output_dir)
    return batch_results

def visualize_batch_results(batch_results, output_dir):
    import matplotlib.pyplot as plt
    field_angles = batch_results['field_angles']
    plot_data = [
        ('Centroid Error vs Field Angle', 'Centroid Error (pixels)', batch_results['centroid_errors'], batch_results['centroid_stds'], 'blue', "centroid_error_vs_angle.png"),
        ('Bearing Vector Error vs Field Angle', 'Angular Error (arcsec)', batch_results['vector_errors'], batch_results['vector_stds'], 'green', "vector_error_vs_angle.png"),
        ('Success Rate vs Field Angle', 'Success Rate', batch_results['success_rates'], None, 'purple', "success_rate_vs_angle.png")
    ]
    for title, ylabel, data, stds, color, filename in plot_data:
        plt.figure(figsize=(10, 6))
        if stds: plt.errorbar(field_angles, data, yerr=stds, fmt='o-', color=color, capsize=5)
        else: plt.plot(field_angles, data, 'o-', color=color)
        plt.title(title); plt.xlabel('Field Angle (degrees)'); plt.ylabel(ylabel)
        if title.startswith('Success Rate'): plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300); plt.close()
        
    # Correcting DataFrame column names to match batch_results keys
    df_data = {
        'Field_Angle': batch_results['field_angles'],
        'Centroid_Error_Px': batch_results['centroid_errors'],
        'Centroid_Error_Std': batch_results['centroid_stds'],
        'Vector_Error_Arcsec': batch_results['vector_errors'],
        'Vector_Error_Std': batch_results['vector_stds'],
        'Success_Rate': batch_results['success_rates']
    }
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, "field_angle_results.csv"), index=False)
    logger.info(f"Saved batch results to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Star Tracker Centroiding and Bearing Vector Pipeline")
    parser.add_argument("--psf", required=True, help="Path to PSF file or directory (e.g., PSF_sims/Gen_1)")
    parser.add_argument("--pattern", default="*_deg.txt", help="File pattern for directory (default: *_deg.txt)") # Updated default
    mag_group = parser.add_mutually_exclusive_group()
    mag_group.add_argument("--magnitude", type=float, default=3.0, help="Star magnitude")
    mag_group.add_argument("--direct-photons", type=float, dest="photon_count", help="Direct photon count")
    parser.add_argument("--f-stop", type=float, default=1.7, help="Optical f-stop")
    parser.add_argument("--focal-length", type=float, default=32.0, dest="config_focal_length", help="Focal length (mm) for camera configuration") # Renamed to avoid conflict
    parser.add_argument("--pixel-size", type=float, default=5.5, help="Pixel size (microns)")
    parser.add_argument("--int-time", type=float, default=17.0, help="Integration time (ms)")
    parser.add_argument("--temperature", type=float, default=20.0, help="Temperature (°C)")
    parser.add_argument("--simulations", type=int, default=50, help="Number of Monte Carlo simulations")
    
    analysis_mode_group = parser.add_mutually_exclusive_group()
    analysis_mode_group.add_argument("--mag-sweep", action="store_true", help="Perform magnitude sweep")
    analysis_mode_group.add_argument("--focal-length-analysis", action="store_true", help="Analyze effect of focal length")
    # Add other analysis modes flags here (e.g., --temp-sweep, --f-stop-analysis, --pixel-size-analysis)

    # Arguments for magnitude sweep
    parser.add_argument("--min-mag", type=float, default=0.0, help="Min magnitude for sweep")
    parser.add_argument("--max-mag", type=float, default=6.0, help="Max magnitude for sweep")
    parser.add_argument("--mag-points", type=int, default=13, help="Number of points for magnitude sweep")

    # Arguments for focal length analysis
    parser.add_argument("--focal-lengths", type=float, nargs="+", default=[20, 24, 28, 32, 36, 40, 50], help="Focal lengths to test in mm for analysis")
    # Add arguments for other analysis types here (e.g. --f-stops, --pixel-sizes, --temperatures, --temp-points)


    parser.add_argument("--output", dest="output_dir", default="pipeline_results", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    pipeline = StarTrackerPipeline(debug=args.debug)
    # Use the renamed --focal-length from parser for initial camera config
    pipeline.update_optical_parameters(f_stop=args.f_stop, focal_length=args.config_focal_length) 
    pipeline.camera.fpa.pitch = args.pixel_size
    pipeline.scene.int_time = args.int_time
    pipeline.scene.temp = args.temperature
    
    start_time = time.time()
    try:
        if os.path.isfile(args.psf):
            # --- SINGLE PSF FILE PROCESSING ---
            output_base_dir = args.output_dir
            if args.mag_sweep:
                magnitudes = np.linspace(args.min_mag, args.max_mag, args.mag_points)
                output_dir_mag_sweep = os.path.join(output_base_dir, "magnitude_sweep_single_psf")
                metadata, intensity_data = parse_psf_file(args.psf)
                single_psf_data = {metadata.get('field_angle', 0.0): {'metadata': metadata, 'intensity_data': intensity_data, 'file_path': args.psf}}
                pipeline.run_magnitude_sweep(magnitudes=magnitudes, psf_data=single_psf_data, num_trials=args.simulations, output_dir=output_dir_mag_sweep)
                logger.info("Magnitude sweep completed (single PSF).")
            
            elif args.focal_length_analysis:
                output_dir_focal = os.path.join(output_base_dir, "focal_length_analysis_single_psf")
                metadata, intensity_data = parse_psf_file(args.psf)
                single_psf_data = {metadata.get('field_angle', 0.0): {'metadata': metadata, 'intensity_data': intensity_data, 'file_path': args.psf}}
                pipeline.run_optical_parameter_analysis(
                    param_name='focal_length',
                    param_values=args.focal_lengths, # This uses the --focal-lengths list
                    magnitude=args.magnitude,
                    photon_count=args.photon_count,
                    num_trials=args.simulations,
                    psf_data=single_psf_data, # Use the single loaded PSF
                    output_dir=output_dir_focal
                )
                logger.info("Focal length analysis completed (single PSF).")

            # Add elif blocks here for other analysis modes on a single PSF file
            # e.g., elif args.f_stop_analysis: ...
            # e.g., elif args.temp_sweep: ...
            # e.g., elif args.pixel_size_analysis: ...

            else: # Default processing for a single PSF file (no specific analysis mode)
                output_dir_single = os.path.join(output_base_dir, f"single_{os.path.splitext(os.path.basename(args.psf))[0]}")
                single_results = process_single_psf(pipeline, args.psf, magnitude=args.magnitude, photon_count=args.photon_count, num_trials=args.simulations, output_dir=output_dir_single)
                if single_results and single_results['successful_trials'] > 0:
                    logger.info(f"Centroid Error: {single_results['mean_centroid_error_px']:.3f} ± {single_results['std_centroid_error_px']:.3f} px")
                    if not np.isnan(single_results['mean_vector_error_arcsec']):
                         logger.info(f"Vector Error: {single_results['mean_vector_error_arcsec']:.2f} ± {single_results['std_vector_error_arcsec']:.2f} arcsec")

        else: # --- DIRECTORY PROCESSING ---
            output_base_dir = args.output_dir
            all_psf_data = pipeline.load_psf_data(args.psf, args.pattern)
            if not all_psf_data: 
                logger.error(f"No PSF files found in directory {args.psf} with pattern {args.pattern}. Exiting."); 
                return

            if args.mag_sweep:
                magnitudes = np.linspace(args.min_mag, args.max_mag, args.mag_points)
                output_dir_mag_sweep_all = os.path.join(output_base_dir, "magnitude_sweep_on_axis_psf_from_dir")
                # run_magnitude_sweep uses the on-axis (or first) PSF from the loaded all_psf_data
                pipeline.run_magnitude_sweep(magnitudes=magnitudes, psf_data=all_psf_data, num_trials=args.simulations, output_dir=output_dir_mag_sweep_all)
                logger.info("Magnitude sweep completed (using on-axis PSF from directory).")

            elif args.focal_length_analysis:
                output_dir_focal_all = os.path.join(output_base_dir, "focal_length_analysis_on_axis_psf_from_dir")
                # run_optical_parameter_analysis also uses the on-axis (or first) PSF
                pipeline.run_optical_parameter_analysis(
                    param_name='focal_length',
                    param_values=args.focal_lengths,
                    magnitude=args.magnitude,
                    photon_count=args.photon_count,
                    num_trials=args.simulations,
                    psf_data=all_psf_data, 
                    output_dir=output_dir_focal_all
                )
                logger.info("Focal length analysis completed (using on-axis PSF from directory).")
            
            # Add elif blocks here for other analysis modes on directory (typically using on-axis PSF)
            # e.g., elif args.f_stop_analysis: ...
            # e.g., elif args.temp_sweep: ...
            # e.g., elif args.pixel_size_analysis: ...

            else: # Default batch processing for directory (analyzes each PSF file for field angle dependency)
                output_dir_batch = os.path.join(output_base_dir, "batch_field_angles")
                batch_results_data = batch_process_psf_files(pipeline, args.psf, pattern=args.pattern, magnitude=args.magnitude, photon_count=args.photon_count, num_trials=args.simulations, output_dir=output_dir_batch)
                if batch_results_data and batch_results_data['field_angles']:
                    logger.info(f"Processed {len(batch_results_data['field_angles'])} field angles.")
                    def safe_mean(data_list):
                        valid_data = [x for x in data_list if isinstance(x, (int, float)) and not np.isnan(x)]
                        return np.mean(valid_data) if valid_data else float('nan')
                    
                    logger.info(f"Average Centroid Error: {safe_mean(batch_results_data['centroid_errors']):.3f} pixels")
                    logger.info(f"Average Vector Error: {safe_mean(batch_results_data['vector_errors']):.2f} arcsec")
                    logger.info(f"Average Success Rate: {safe_mean(batch_results_data['success_rates']):.2f}")

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True) 
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
