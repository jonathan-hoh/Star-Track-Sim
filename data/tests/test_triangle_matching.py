import unittest
import numpy as np
from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.synthetic_catalog import SyntheticCatalogBuilder
from src.multi_star.multi_star_pipeline import MultiStarPipeline
from src.multi_star.validation import validate_triangle_matches

class TestTriangleMatching(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.pipeline = StarTrackerPipeline()
        self.multi_star_pipeline = MultiStarPipeline(self.pipeline)
        self.catalog_builder = SyntheticCatalogBuilder()
        # Load a 0-degree PSF for testing
        self.psf_data = self.pipeline.load_psf_data("PSF_sims/Gen_1", "*0_deg*")[0.0]

    def test_3_star_triangle_matching(self):
        """Test the full pipeline for a 3-star triangle matching scenario."""
        # 1. Create 3-star catalog
        catalog = self.catalog_builder.create_triangle_catalog()

        # 2. Run the multi-star analysis
        results = self.multi_star_pipeline.run_multi_star_analysis(catalog, self.psf_data)

        # 3. Validate the results
        self.assertEqual(results['detected_stars'], 3, "Should detect 3 stars")
        self.assertEqual(len(results['star_matches']), 1, "Should find 1 triangle match")
        self.assertEqual(results['validation']['status'], 'passed', "Validation should pass")

if __name__ == '__main__':
    unittest.main()