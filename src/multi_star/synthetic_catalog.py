import numpy as np
import pandas as pd
from ..BAST.catalog import Catalog

class SyntheticCatalogBuilder:
    """Creates synthetic star catalogs compatible with BAST.catalog.Catalog"""

    def generate_catalog(
        self,
        num_in_fov: int,
        num_out_fov: int,
        fov_deg: float,
        distribution: str = 'spread',
        magnitude_range: tuple = (2.0, 5.0),
        cluster_center_ra_deg: float = 0.0,
        cluster_center_dec_deg: float = 0.0,
        cluster_std_dev_deg: float = 1.0
    ):
        """
        Generates a flexible synthetic star catalog.

        Args:
            num_in_fov (int): Number of stars to generate within the FOV.
            num_out_fov (int): Number of "red herring" stars to generate outside the FOV.
            fov_deg (float): The full field of view in degrees.
            distribution (str, optional): Star distribution ('spread' or 'clustered').
                                        Defaults to 'spread'.
            magnitude_range (tuple, optional): Range of star magnitudes (min, max).
                                             Defaults to (2.0, 5.0).
            cluster_center_ra_deg (float, optional): Center RA for 'clustered' mode.
                                                  Defaults to 0.0.
            cluster_center_dec_deg (float, optional): Center Dec for 'clustered' mode.
                                                   Defaults to 0.0.
            cluster_std_dev_deg (float, optional): Std dev for 'clustered' mode.
                                                Defaults to 1.0.

        Returns:
            Catalog: A BAST-compatible catalog object.
        """
        in_fov_stars = self._generate_in_fov_stars(
            num_in_fov, fov_deg, distribution, magnitude_range,
            cluster_center_ra_deg, cluster_center_dec_deg, cluster_std_dev_deg
        )

        out_fov_stars = self._generate_out_fov_stars(
            num_out_fov, fov_deg, magnitude_range
        )

        all_stars = pd.concat([in_fov_stars, out_fov_stars], ignore_index=True)
        all_stars['Star ID'] = range(1, len(all_stars) + 1)

        # Use existing BAST catalog - it will auto-compute triplets!
        # The FOV for the catalog object should be larger to include out-of-fov stars for matching
        catalog_fov = fov_deg * 3
        catalog = Catalog(all_stars, fov=catalog_fov)
        return catalog

    def _generate_in_fov_stars(
        self, num_stars, fov_deg, distribution, magnitude_range,
        center_ra_deg, center_dec_deg, std_dev_deg
    ):
        if num_stars == 0:
            return pd.DataFrame(columns=['RA', 'DE', 'Magnitude'])

        half_fov_rad = np.deg2rad(fov_deg / 2)

        if distribution == 'spread':
            # Uniformly spread stars across the FOV
            ra_rad = np.random.uniform(-half_fov_rad, half_fov_rad, num_stars)
            de_rad = np.random.uniform(-half_fov_rad, half_fov_rad, num_stars)
        elif distribution == 'clustered':
            # Cluster stars around a central point using a normal distribution
            center_ra_rad = np.deg2rad(center_ra_deg)
            center_de_rad = np.deg2rad(center_dec_deg)
            std_dev_rad = np.deg2rad(std_dev_deg)
            ra_rad = np.random.normal(center_ra_rad, std_dev_rad, num_stars)
            de_rad = np.random.normal(center_de_rad, std_dev_rad, num_stars)
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

        magnitudes = np.random.uniform(magnitude_range[0], magnitude_range[1], num_stars)

        return pd.DataFrame({'RA': ra_rad, 'DE': de_rad, 'Magnitude': magnitudes})

    def _generate_out_fov_stars(self, num_stars, fov_deg, magnitude_range):
        if num_stars == 0:
            return pd.DataFrame(columns=['RA', 'DE', 'Magnitude'])

        # Generate "red herring" stars in a ring just outside the FOV
        # from fov_deg/2 to fov_deg * 1.5 from the center
        min_radius_rad = np.deg2rad(fov_deg / 2) * 1.1 # 10% buffer
        max_radius_rad = np.deg2rad(fov_deg * 1.5)

        # Generate random points in polar coordinates and convert to Cartesian
        radius = np.sqrt(np.random.uniform(min_radius_rad**2, max_radius_rad**2, num_stars))
        angle = np.random.uniform(0, 2 * np.pi, num_stars)
        
        ra_rad = radius * np.cos(angle)
        de_rad = radius * np.sin(angle)
        
        magnitudes = np.random.uniform(magnitude_range[0], magnitude_range[1], num_stars)

        return pd.DataFrame({'RA': ra_rad, 'DE': de_rad, 'Magnitude': magnitudes})


    def create_triangle_catalog(self, separation_deg=5.0):
        """[DEPRECATED] 3 stars for basic triangle matching. Use generate_catalog instead."""
        # Create stars in sky coordinates (RA, Dec)
        stars_data = {
            'Star ID': [1, 2, 3],
            'RA': [0.0, np.deg2rad(separation_deg), np.deg2rad(separation_deg / 2)],
            'DE': [0.0, 0.0, np.deg2rad(separation_deg * 0.866)],  # Equilateral triangle
            'Magnitude': [3.0, 3.0, 3.0]
        }
        df = pd.DataFrame(stars_data)
        # Use existing BAST catalog - it will auto-compute triplets!
        catalog = Catalog(df, fov=10.0)
        return catalog

    def create_pyramid_catalog(self, separation_deg=5.0):
        """[DEPRECATED] 4 stars for pyramid validation. Use generate_catalog instead."""
        stars_data = {
            'Star ID': [1, 2, 3, 4],
            'RA': [0.0, np.deg2rad(separation_deg), np.deg2rad(separation_deg), 0.0],
            'DE': [0.0, 0.0, np.deg2rad(separation_deg), np.deg2rad(separation_deg)],  # Square
            'Magnitude': [3.0, 3.0, 3.0, 3.0]
        }
        df = pd.DataFrame(stars_data)
        catalog = Catalog(df, fov=10.0)
        return catalog