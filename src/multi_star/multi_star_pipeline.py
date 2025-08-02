from ..core.star_tracker_pipeline import StarTrackerPipeline
from ..BAST.catalog import Catalog
from ..BAST.match import match
from .scene_generator import MultiStarSceneGenerator
from .multi_star_radiometry import MultiStarRadiometry

class MultiStarPipeline:
    """Extends existing pipeline for multi-star scenes"""

    def __init__(self, pipeline: StarTrackerPipeline):
        self.pipeline = pipeline
        self.scene_generator = MultiStarSceneGenerator(pipeline)
        self.radiometry = MultiStarRadiometry(pipeline)

    def run_multi_star_analysis(self, synthetic_catalog: Catalog, psf_data):
        """Complete multi-star analysis workflow"""

        # 1. Generate scene (new)
        scene_data = self.scene_generator.generate_scene(synthetic_catalog)

        # 2. Render scene (new)
        scene_data = self.radiometry.render_scene(scene_data, psf_data)

        # 3. Detect stars (existing pipeline)
        centroid_results = self.pipeline.detect_stars_and_calculate_centroids(
            [scene_data['detector_image']], threshold_sigma=4.0
        )

        # 4. Calculate bearing vectors (existing pipeline)
        bearing_results = self.pipeline.calculate_bearing_vectors(
            centroid_results['centroids'],
            scene_data['detector_image'].shape,
            sensor_pixel_pitch_um=0.5  # 0° PSF scale
        )

        # 5. Triangle matching (existing BAST)
        star_matches = match(
            observed_stars=bearing_results['bearing_vectors'],
            catalog=synthetic_catalog
        )

        # 6. Validation (new)
        validation_results = self.validate_matches(star_matches, scene_data)

        return {
            'scene_data': scene_data,
            'detected_stars': len(centroid_results['centroids']),
            'star_matches': star_matches,
            'validation': validation_results
        }

    def validate_matches(self, star_matches, scene_data):
        """
        Validate the matches based on the ground truth from the scene data.
        """
        from .validation import validate_triangle_matches, validate_pyramid_consistency

        num_stars = len(scene_data['stars'])
        ground_truth = scene_data['ground_truth']

        if num_stars == 3:
            return validate_triangle_matches(star_matches, ground_truth)
        elif num_stars == 4:
            return validate_pyramid_consistency(star_matches, ground_truth)
        else:
            return {"status": "unknown", "reason": f"No validation logic for {num_stars} stars."}