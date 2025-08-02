import numpy as np
from ..core.star_tracker_pipeline import StarTrackerPipeline
from ..BAST.catalog import Catalog

class MultiStarSceneGenerator:
    """Generates detector scenes from synthetic catalogs"""

    def __init__(self, pipeline: StarTrackerPipeline):
        self.pipeline = pipeline

    def generate_scene(self, synthetic_catalog: Catalog,
                      detector_center_ra=0.0, detector_center_dec=0.0):
        """Create multi-star detector scene from catalog"""

        scene_data = {
            'stars': [],
            'detector_image': None,
            'ground_truth': {
                'catalog': synthetic_catalog,
                'expected_triangles': self._extract_expected_triangles(synthetic_catalog)
            }
        }

        # Convert RA/Dec to detector positions
        for idx, star_row in synthetic_catalog.iterrows():
            detector_pos = self._sky_to_detector(
                star_row['RA'], star_row['DE'],
                detector_center_ra, detector_center_dec
            )
            scene_data['stars'].append({
                'catalog_idx': idx,
                'detector_position': detector_pos,
                'magnitude': star_row['Magnitude'],
                'star_id': star_row['Star ID']
            })

        return scene_data

    def _sky_to_detector(self, ra, dec, center_ra, center_dec):
        """Convert RA/Dec to detector pixel coordinates for CMV4000"""
        # Use CMV4000 specifications: 2048x2048, 5.5μm pixel pitch
        detector_size = 2048
        pixel_pitch_um = 5.5
        
        # Use the camera's focal length for realistic projection
        focal_length_mm = self.pipeline.camera.f_length  # Focal length in mm
        focal_length_um = focal_length_mm * 1000  # Convert to microns
        
        # Calculate pixels per radian based on focal length and pixel pitch
        pixels_per_radian = focal_length_um / pixel_pitch_um
        
        # Angular offsets from detector center (small angle approximation)
        delta_ra = ra - center_ra
        delta_dec = dec - center_dec

        # Convert to detector coordinates relative to center
        x_offset = pixels_per_radian * delta_ra
        y_offset = pixels_per_radian * delta_dec
        
        # Place relative to detector center
        x_detector = detector_size // 2 + x_offset
        y_detector = detector_size // 2 + y_offset

        return (x_detector, y_detector)

    def _extract_expected_triangles(self, catalog):
        # This is a placeholder for now.
        # In a real scenario, you'd extract the pre-computed triangles from the catalog.
        if "Triplets" in catalog.columns:
            return catalog["Triplets"].dropna().tolist()
        return []