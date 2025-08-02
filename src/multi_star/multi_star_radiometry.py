import numpy as np
from ..core.star_tracker_pipeline import StarTrackerPipeline

class MultiStarRadiometry:
    """Creates detector images with multiple PSFs"""

    def __init__(self, pipeline: StarTrackerPipeline):
        self.pipeline = pipeline

    def render_scene(self, scene_data, psf_data):
        """Render multiple stars onto detector using existing PSF"""

        # Create full CMV4000 detector size
        detector_image = np.zeros((2048, 2048))

        for star in scene_data['stars']:
            # Use existing FPA-projected simulation method
            star_sim = self.pipeline.simulate_star(magnitude=star['magnitude'])
            
            # Run FPA projection simulation for single star 
            fpa_results = self.pipeline.run_monte_carlo_simulation_fpa_projected(
                psf_data, 
                photon_count=star_sim['photon_count'],
                num_trials=1,
                target_pixel_pitch_um=5.5,
                create_full_fpa=False
            )
            
            # Get the simulated FPA PSF image
            star_psf_image = fpa_results['projection_results']['simulations'][0]

            # Place at star's detector position
            detector_image = self._place_psf_at_position(
                detector_image,
                star_psf_image,
                star['detector_position']
            )

        scene_data['detector_image'] = detector_image
        return scene_data

    def _place_psf_at_position(self, detector_image, psf_image, position):
        """Place PSF at specified detector position with proper bounds handling"""
        import logging
        logger = logging.getLogger(__name__)
        
        psf_height, psf_width = psf_image.shape
        det_height, det_width = detector_image.shape

        # Calculate placement bounds
        center_x, center_y = position
        psf_start_x = int(center_x - psf_width // 2)
        psf_start_y = int(center_y - psf_height // 2)
        
        # Calculate overlap regions
        det_start_x = max(0, psf_start_x)
        det_start_y = max(0, psf_start_y)
        det_end_x = min(det_width, psf_start_x + psf_width)
        det_end_y = min(det_height, psf_start_y + psf_height)
        
        # Calculate corresponding PSF regions
        psf_offset_x = det_start_x - psf_start_x
        psf_offset_y = det_start_y - psf_start_y
        psf_end_x = psf_offset_x + (det_end_x - det_start_x)
        psf_end_y = psf_offset_y + (det_end_y - det_start_y)
        
        # Check if there's any overlap
        if det_start_x >= det_end_x or det_start_y >= det_end_y:
            logger.warning(f"PSF at position ({center_x:.1f}, {center_y:.1f}) is completely outside detector bounds")
            return detector_image
        
        # Place the overlapping portion
        try:
            detector_image[det_start_y:det_end_y, det_start_x:det_end_x] += \
                psf_image[psf_offset_y:psf_end_y, psf_offset_x:psf_end_x]
            
            logger.info(f"Placed PSF at ({center_x:.1f}, {center_y:.1f}), detector region: ({det_start_x}, {det_start_y}) to ({det_end_x}, {det_end_y})")
        except Exception as e:
            logger.error(f"Failed to place PSF at ({center_x:.1f}, {center_y:.1f}): {e}")

        return detector_image