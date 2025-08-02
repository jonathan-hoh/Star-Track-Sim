import logging
import multiprocessing as mp
from itertools import combinations
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

CATALOG_DIR = Path(__file__).parent / "catalogs"

logger = logging.getLogger(__name__)


def triplet_worker(task):
    """
    Processes a list of triplets for a star

    Args:
        task: dict
            {
                "index": int,               # Catalog index
                "star": SkyCoord,           # Primary star coordiante
                "neighbors":  SkyCoord[]    # FOV neighbors of star
            }

    Returns:
        index, triplets
    """
    idx = task["index"]
    star = task["star"]
    neighbors = task["neighbors"]
    triplets = []

    # Combinations of two for all neighbors + star = triplet
    combs = combinations(neighbors, 2)
    for group in combs:
        # Star is always 0th angle in triplet
        # Group stars are arranged against star
        triplets.append(
            (
                star.separation(group[0]).deg,
                star.separation(group[1]).deg,
                group[0].separation(group[1]).deg,
            )
        )
    return idx, triplets


def prepare_tasks(stars, fov_radius_rad):
    """
    Prepares tripleting tasks for parallel processing

    Args:
        stars: SkyCoord[] star catalog
        fov_radius_rad: field of view radius (radians)

    Returns:
        tasks to pass to triplet_worker() via mp.Pool.imap()
    """
    # First, find all "neighbors" within FOV of each star
    tasks = []
    # Pick a star from catalog
    for idx, star in enumerate(stars):
        # Calculate separation to all other stars
        separations = star.separation(stars)
        # Downselect by separation that would exist in our FOV that isn't the star itself
        filtered_catalog = stars[
            (separations < (fov_radius_rad * u.deg)) & (separations != 0 * u.deg)
        ]

        # Append list to tasks
        tasks.append({"index": idx, "star": star, "neighbors": filtered_catalog})

    return tasks


def calculate_catalog_triplets(df: pd.DataFrame, fov_radius_rad: float) -> pd.DataFrame:
    """
    Add triplets to star catalog

    Args:
        star_df: Star catalog DataFrame
        fov_radius_rad: field of view radius (radians)

    Returns:
        Updated star catalog DataFrame
    """
    # Create SkyCoord objects using the RA and Dec from the DataFrame
    stars = SkyCoord(
        ra=df["RA"].values * u.rad,
        dec=df["DE"].values * u.rad,
        frame="icrs",
    )

    # First, find all "neighbors" within FOV of each star
    tasks = prepare_tasks(stars, fov_radius_rad)

    # Second, calculate all tripltes for those neighbors
    df["Triplets"] = None  # Create column for triplet storage
    with mp.Pool(mp.cpu_count()) as pool:
        total = len(tasks)
        # Optional progress bar for debugging
        with tqdm(
            total=total,
            desc="Processing triplets",
            unit="group",
            disable=not logger.isEnabledFor(logging.DEBUG),
        ) as pbar:
            # Update catalog with triplet angles
            for idx, triplets in pool.imap(triplet_worker, tasks):
                if len(triplets) > 0:
                    df.at[df.index[idx], "Triplets"] = triplets
                pbar.update()

    return df


class Catalog(pd.DataFrame):
    input_columns = ("Star ID", "RA", "DE", "Magnitude")
    _fov = None

    # Required to ensure operations return MyDataFrame, not base DataFrame
    @property
    def _constructor(self):

        # Contructor to tolerate pd.DataFrame extension properties
        # Gets called when doing complex filtering with pandas
        def _c(*args, **kwargs):
            # Pass current FOV into new DataFrame
            return Catalog(*args, self._fov, **kwargs)

        return _c

    def __init__(self, df, fov, *args, **kwargs):
        """
        Initialize catalog object

        Args:
            df: Pandas dataframe with (Star ID, RA, DE, Magnitude) columns
            fov: Field of view (degrees)
        """
        if not all(col in df.columns for col in self.input_columns):
            raise ValueError(f"Invalid source catalog, must have {self.input_columns}")

        self._fov = fov

        super().__init__(*(df, *args), **kwargs)
        self._make_combinations(fov)

    def _make_combinations(self, fov):
        if "Triplets" not in self.columns:
            calculate_catalog_triplets(self, fov)
            logging.debug("Triplets successfully added to catalog")
        else:
            logging.debug("Catalog already contains triplets")

    def num_triplets(self):
        """
        Count the total number of triplets in the catalog

        Returns:
            Total number of triplets in catalog
        """
        return int(self["Triplets"].dropna().apply(len).sum())

    def save(self, filename):
        """Save the catalog to file."""
        self.to_pickle(filename)
        logging.debug(f"Catalog saved to {filename}")


def from_csv(filename, magnitude, fov):
    """
    Create a new Catalog object from a CSV.

    CSV must have heading row with the following values:
    "Star ID", "RA", "DE", "Magnitude"

    RA and DE are assumed to be in radians
    Magnitude is assumed to be in mV

    The returned catalog will be filtered by provided magnitude threshold.
    The returned catalog will also calculate all triplets that exist within the provided field of view (fov)

    Args:
        filename: Path to CSV
        magnitude: Magnitude threshold for catalog
        fov: Field of view for catalog (half angle, degrees)

    Returns:
        Catalog object
    """
    df = pd.read_csv(filename)

    # Clean up input data
    cols = [col for col in Catalog.input_columns if col in df.columns]
    for col in df.columns:
        if col not in cols:
            df.drop(columns=col, inplace=True)

    logging.debug(f"Filtering catalog to magnitudes below {magnitude}")
    filtered_df = df.loc[df["Magnitude"] <= magnitude].copy()

    logging.debug(f"Generating catalog object")
    return Catalog(filtered_df, fov)


def from_pickle(filename):
    """
    Load a catalog from a pickle file

    Args:
        filename: Path to pickle flie

    Raises:
        ValueError: Raised if pickle file is not a valid Catalog object

    Returns:
        Catalog object
    """
    df = pd.read_pickle(filename)

    if not isinstance(df, Catalog):
        raise ValueError(f"{filename} not a valid catalog")

    return df


if __name__ == "__main__":
    import logging

    logging.getLogger("bast").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    magnitude = 2
    fov = 10

    c = from_csv(CATALOG_DIR / "HIPPARCOS.csv", magnitude, fov)

    print(c)

    c.save(f"HIPPARCOS_{magnitude}_{fov}.bast")

    print("done")
