#!/usr/bin/env python3
"""
Multi-Star Pipeline Visualization Script

This script creates comprehensive visualizations of the multi-star BAST matching pipeline
to help explain the system to team members. It walks through each major stage with 
clear, educational plots.

Usage:
    python3 multi_star_visualization.py

Created for team education and system understanding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import sys
import os
import argparse
from pathlib import Path
from itertools import combinations

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.star_tracker_pipeline import StarTrackerPipeline
from src.multi_star.multi_star_pipeline import MultiStarPipeline
from src.multi_star.peak_detection import detect_stars_peak_method
from src.BAST.match import calculate_vector_angle
from src.BAST.catalog import Catalog  # Import Catalog class

# Configure logging to see debug information
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

class MultiStarVisualizer:
    """Creates educational visualizations of the multi-star pipeline"""
    
    def __init__(self, catalog_path: Path):
        """Initialize the visualizer with pipeline components"""
        print("Initializing Multi-Star Pipeline Visualizer...")
        
        self.catalog_path = catalog_path
        self.output_dir = self._setup_output_directory()
        
        # Initialize pipeline components
        self.pipeline = StarTrackerPipeline()
        self.multi_star_pipeline = MultiStarPipeline(self.pipeline)
        
        # Load data
        self.psf_data = self._load_psf_data()
        self.catalog = self._load_catalog_from_file()
        
        # Run simulation pipeline
        self._run_simulation()
        
        print("Visualization setup complete!")

    def _setup_output_directory(self):
        """Create and return the output directory path."""
        output_dir = Path(f"outputs/multi_visualize_outputs/{self.catalog_path.stem}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Outputting visualizations to: {output_dir.resolve()}")
        return output_dir

    def _load_psf_data(self):
        """Load the default PSF data."""
        print("Loading PSF data...")
        psf_data_dict = self.pipeline.load_psf_data("data/PSF_sims/Gen_1", "*0_deg*")
        if not psf_data_dict:
            raise ValueError("No PSF files found. Check PSF_sims/Gen_1/ directory.")
        return list(psf_data_dict.values())[0]

    def _load_catalog_from_file(self):
        """Load a BAST-compatible catalog from a CSV file."""
        print(f"Loading catalog from: {self.catalog_path}")
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")
        
        df = pd.read_csv(self.catalog_path)
        # The BAST Catalog class expects a fov parameter. We'll use a standard 10 deg.
        return Catalog(df, fov=10.0)

    def _run_simulation(self):
        """Run the core multi-star simulation pipeline."""
        # Generate and render scene
        print("Generating multi-star scene...")
        self.scene_data = self.multi_star_pipeline.scene_generator.generate_scene(self.catalog)
        
        # Add RA/Dec to scene data for validation
        for i, star in enumerate(self.scene_data['stars']):
            catalog_row = self.catalog.iloc[star['catalog_idx']]
            star['ra'] = catalog_row['RA']
            star['dec'] = catalog_row['DE']
        
        # Render scene
        self.scene_data = self.multi_star_pipeline.radiometry.render_scene(self.scene_data, self.psf_data)
        
        # Detect stars
        print("Detecting stars...")
        self.centroids = detect_stars_peak_method(
            self.scene_data['detector_image'],
            min_intensity_fraction=0.1,
            min_separation=20 # Reduced for potentially denser fields
        )
        
        # Calculate bearing vectors
        print("Calculating bearing vectors...")
        bearing_results = self.pipeline.calculate_bearing_vectors(
            self.centroids,
            self.scene_data['detector_image'].shape,
            sensor_pixel_pitch_um=5.5  # CMV4000 pixel pitch
        )
        self.bearing_vectors = bearing_results['bearing_vectors']
        
    def create_single_star_psf_comparison(self):
        """
        Visualization 1: Single-star PSF in original plane vs FPA projection
        Shows the difference between high-res simulation and detector reality
        """
        print("\nCreating Visualization 1: Single-Star PSF Comparison...")
        
        # Get original PSF data
        original_psf = self.psf_data['intensity_data']
        original_metadata = self.psf_data['metadata']
        
        # Project to FPA for comparison
        fpa_psf = self.pipeline.project_psf_to_fpa_grid(
            original_psf,
            original_metadata.get('data_spacing', 0.5),
            5.5  # CMV4000 pixel pitch
        )
        
        # Calculate true centroids
        original_true_center = self.pipeline.calculate_true_psf_centroid(
            original_psf, original_metadata.get('data_spacing', 0.5)
        )
        
        fpa_true_center = self.pipeline.calculate_true_psf_centroid(
            fpa_psf['projected_psf'], 5.5
        )
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Single-Star PSF: Original vs FPA Projection', fontsize=16, fontweight='bold')
        
        # Original PSF
        im1 = ax1.imshow(original_psf, cmap='hot', origin='lower')
        ax1.plot(original_true_center[0]/0.5, original_true_center[1]/0.5, 'cyan', marker='x', markersize=12, markeredgewidth=3, label='True Center')
        ax1.set_title('Original PSF (128×128, 0.5μm/pixel)')
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Pixels')
        ax1.legend()
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # FPA Projected PSF
        im2 = ax2.imshow(fpa_psf['projected_psf'], cmap='hot', origin='lower')
        ax2.plot(fpa_true_center[0]/5.5, fpa_true_center[1]/5.5, 'cyan', marker='x', markersize=12, markeredgewidth=3, label='True Center')
        ax2.set_title('FPA Projected PSF (11×11, 5.5μm/pixel)')
        ax2.set_xlabel('CMV4000 Pixels')
        ax2.set_ylabel('CMV4000 Pixels')
        ax2.legend()
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Multi-star detector image for context
        im3 = ax3.imshow(self.scene_data['detector_image'], cmap='hot', origin='lower')
        # Plot detected centroids
        # Dynamically generate colors for all centroids
        colors = plt.cm.jet(np.linspace(0, 1, len(self.centroids)))
        for i, centroid in enumerate(self.centroids):
            ax3.plot(centroid[0], centroid[1], color=colors[i], marker='+',
                     markersize=12, markeredgewidth=2, label=f'Detected Star {i+1}')
        ax3.set_title('Multi-Star Detector Image (For Context)')
        ax3.set_xlabel('Detector Pixels')
        ax3.set_ylabel('Detector Pixels')
        ax3.legend()
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Intensity profiles comparison
        center_row_orig = original_psf[original_psf.shape[0]//2, :]
        center_row_fpa = fpa_psf['projected_psf'][fpa_psf['projected_psf'].shape[0]//2, :]
        
        x_orig = np.arange(len(center_row_orig)) * 0.5  # Convert to microns
        x_fpa = np.arange(len(center_row_fpa)) * 5.5   # Convert to microns
        
        ax4.plot(x_orig, center_row_orig, 'b-', linewidth=2, label='Original PSF (0.5μm/px)')
        ax4.plot(x_fpa, center_row_fpa, 'r-', linewidth=2, marker='o', markersize=4, label='FPA Projected (5.5μm/px)')
        ax4.set_title('PSF Intensity Profiles (Center Row)')
        ax4.set_xlabel('Position (μm)')
        ax4.set_ylabel('Normalized Intensity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'visualization_1_single_star_psf_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
        
    def create_multi_star_scene_visualization(self):
        """
        Visualization 2: Multi-star scene on FPA with detected centroids
        Shows the complete multi-star simulation with detection results
        """
        print("\nCreating Visualization 2: Multi-Star Scene with Centroids...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Star Scene Analysis', fontsize=16, fontweight='bold')
        
        # Dynamic color scheme for stars
        num_stars_total = len(self.scene_data['stars'])
        true_colors = plt.cm.cool(np.linspace(0, 1, num_stars_total))
        
        num_centroids = len(self.centroids)
        detected_colors = plt.cm.autumn(np.linspace(0, 1, num_centroids))

        # Main detector image with all detected centroids
        im1 = ax1.imshow(self.scene_data['detector_image'], cmap='hot', origin='lower')
        
        # Plot detected centroids
        for i, centroid in enumerate(self.centroids):
            ax1.plot(centroid[0], centroid[1], color=detected_colors[i], marker='+',
                     markersize=15, markeredgewidth=3, label=f'Detected Star {i+1}')
        
        # Plot true star positions
        for i, star in enumerate(self.scene_data['stars']):
            pos = star['detector_position']
            ax1.plot(pos[0], pos[1], color=true_colors[i], marker='x',
                     markersize=12, markeredgewidth=2, alpha=0.7, label=f'True Pos {star["star_id"]}')
        
        ax1.set_title(f'Complete Multi-Star Detector Image ({self.catalog_path.name})')
        ax1.set_xlabel('Detector X (pixels)')
        ax1.set_ylabel('Detector Y (pixels)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Zoomed view of each star - handle variable number of stars
        subplot_axes = [ax2, ax3, ax4]
        star_titles = ['Star 1 (Zoomed)', 'Star 2 (Zoomed)', 'Star 3 (Zoomed)']
        
        # Process up to 3 stars or as many as available
        max_stars = min(3, len(self.scene_data['stars']))
        
        for i in range(max_stars):
            star = self.scene_data['stars'][i]
            ax = subplot_axes[i]
            title = star_titles[i]
            
            # Extract region around star with bounds checking
            pos = star['detector_position']
            window_size = 50
            img_height, img_width = self.scene_data['detector_image'].shape
            
            x_start = max(0, int(pos[0] - window_size))
            x_end = min(img_width, int(pos[0] + window_size))
            y_start = max(0, int(pos[1] - window_size))
            y_end = min(img_height, int(pos[1] + window_size))
            
            # Ensure we have a valid region
            if x_start >= x_end or y_start >= y_end:
                ax.text(0.5, 0.5, 'No valid region', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue
            
            star_region = self.scene_data['detector_image'][y_start:y_end, x_start:x_end]
            
            # Skip if region is empty
            if star_region.size == 0:
                ax.text(0.5, 0.5, 'Empty region', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue
            
            im = ax.imshow(star_region, cmap='hot', origin='lower', 
                          extent=[x_start, x_end, y_start, y_end])
            
            # Plot true position
            ax.plot(pos[0], pos[1], color=true_colors[i], marker='x',
                   markersize=10, markeredgewidth=2, label=f'True Pos {star["star_id"]}')
            
            # Plot detected centroid if available - find closest centroid to this star
            if self.centroids:
                # Find closest centroid to this star position
                distances = [np.sqrt((c[0] - pos[0])**2 + (c[1] - pos[1])**2) for c in self.centroids]
                if distances:
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 100:  # Only plot if reasonably close
                        centroid = self.centroids[closest_idx]
                        ax.plot(centroid[0], centroid[1], color=detected_colors[closest_idx], marker='+',
                               markersize=12, markeredgewidth=2, label='Detected Centroid')
            
            ax.set_title(f'Star {star["star_id"]} (Zoomed)')
            ax.set_xlabel('Detector X (pixels)')
            ax.set_ylabel('Detector Y (pixels)')
            ax.legend()
        
        # Clear unused subplots
        for i in range(max_stars, 3):
            subplot_axes[i].axis('off')
            subplot_axes[i].text(0.5, 0.5, f'Star {i+1} not available', ha='center', va='center', transform=subplot_axes[i].transAxes)
            
        plt.tight_layout()
        output_file = self.output_dir / 'visualization_2_multi_star_scene.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
        
    def create_observed_bearing_vectors_3d(self):
        """
        Visualization 3: 3D bearing vectors from observed stars
        Shows the 3D unit vectors calculated from detected centroids
        """
        print("\nCreating Visualization 3: Observed Bearing Vectors (3D)...")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dynamic color scheme
        num_vectors = len(self.bearing_vectors)
        colors = plt.cm.autumn(np.linspace(0, 1, num_vectors))

        # Plot bearing vectors - handle variable number of vectors
        if not self.bearing_vectors:
            ax.text(0.5, 0.5, 0.5, 'No bearing vectors available', ha='center', va='center', fontsize=14)
            return
            
        for i, bv in enumerate(self.bearing_vectors):
            # Plot vector from origin
            ax.quiver(0, 0, 0, bv[0], bv[1], bv[2],
                     color=colors[i], linewidth=3, alpha=0.8,
                     arrow_length_ratio=0.1, label=f'Observed Vector {i+1}')
            
            # Plot endpoint
            ax.scatter([bv[0]], [bv[1]], [bv[2]],
                      color=colors[i], s=100, alpha=0.8)
            
            # Add vector coordinates as text
            ax.text(bv[0]*1.1, bv[1]*1.1, bv[2]*1.1,
                   f'({bv[0]:.4f}, {bv[1]:.4f}, {bv[2]:.4f})',
                   color=colors[i], fontsize=9)
        
        # Calculate and display inner angles
        angles = []
        if len(self.bearing_vectors) >= 2:
            for i in range(len(self.bearing_vectors)):
                for j in range(i+1, len(self.bearing_vectors)):
                    angle = calculate_vector_angle(self.bearing_vectors[i], self.bearing_vectors[j])
                    angle_deg = np.degrees(angle)
                    angles.append(angle_deg)
                    
                    # Add angle annotation
                    mid_point = (self.bearing_vectors[i] + self.bearing_vectors[j]) / 2
                    mid_point = mid_point / np.linalg.norm(mid_point) * 0.5
                    ax.text(mid_point[0], mid_point[1], mid_point[2], 
                           f'{angle_deg:.3f}°', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Customize plot
        ax.set_xlabel('X (Bearing Vector Component)')
        ax.set_ylabel('Y (Bearing Vector Component)')
        ax.set_zlabel('Z (Bearing Vector Component)')
        ax.set_title('Observed Bearing Vectors from Detected Stars\n(With Inner Angles)', 
                    fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        ax.legend()
        
        # Add summary text
        if angles:
            angle_text = ", ".join([f"{angle:.3f}°" for angle in angles])
            summary_text = f"Inner Angles: {angle_text}"
        else:
            summary_text = "Inner Angles: Insufficient bearing vectors for angle calculation"
        
        ax.text2D(0.02, 0.02, summary_text, transform=ax.transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                 fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / 'visualization_3_observed_bearing_vectors.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
        
    def create_catalog_bearing_vectors_3d(self):
        """
        Visualization 4: 3D bearing vectors from catalog stars
        Shows the expected bearing vectors from synthetic catalog
        """
        print("\nCreating Visualization 4: Catalog Bearing Vectors (3D)...")
        
        # Calculate catalog bearing vectors from RA/Dec
        catalog_bearing_vectors = []
        for star in self.scene_data['stars']:
            # Convert RA/Dec to bearing vector using the same projection method
            ra, dec = star['ra'], star['dec']
            
            # Simple projection (small angle approximation)
            x = ra  # RA offset from center
            y = dec  # Dec offset from center
            z = 1.0  # Optical axis
            
            # Normalize to unit vector
            magnitude = np.sqrt(x*x + y*y + z*z)
            bearing_vector = np.array([x/magnitude, y/magnitude, z/magnitude])
            catalog_bearing_vectors.append(bearing_vector)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dynamic color scheme
        num_vectors = len(catalog_bearing_vectors)
        colors = plt.cm.cool(np.linspace(0, 1, num_vectors))

        # Plot catalog bearing vectors
        if not catalog_bearing_vectors:
            ax.text(0.5, 0.5, 0.5, 'No catalog bearing vectors available', ha='center', va='center', fontsize=14)
            return None, None
            
        for i, (bv, star_data) in enumerate(zip(catalog_bearing_vectors, self.scene_data['stars'])):
            # Plot vector from origin
            ax.quiver(0, 0, 0, bv[0], bv[1], bv[2],
                     color=colors[i], linewidth=3, alpha=0.8,
                     arrow_length_ratio=0.1, label=f'Catalog Star {star_data["star_id"]}')
            
            # Plot endpoint
            ax.scatter([bv[0]], [bv[1]], [bv[2]],
                      color=colors[i], s=100, alpha=0.8)
            
            # Add vector coordinates as text
            ax.text(bv[0]*1.1, bv[1]*1.1, bv[2]*1.1,
                   f'({bv[0]:.4f}, {bv[1]:.4f}, {bv[2]:.4f})',
                   color=colors[i], fontsize=9)
        
        # Calculate and display inner angles
        catalog_angles = []
        if len(catalog_bearing_vectors) >= 2:
            for i in range(len(catalog_bearing_vectors)):
                for j in range(i+1, len(catalog_bearing_vectors)):
                    angle = calculate_vector_angle(catalog_bearing_vectors[i], catalog_bearing_vectors[j])
                    angle_deg = np.degrees(angle)
                    catalog_angles.append(angle_deg)
                    
                    # Add angle annotation
                    mid_point = (catalog_bearing_vectors[i] + catalog_bearing_vectors[j]) / 2
                    mid_point = mid_point / np.linalg.norm(mid_point) * 0.5
                    ax.text(mid_point[0], mid_point[1], mid_point[2], 
                           f'{angle_deg:.3f}°', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Customize plot
        ax.set_xlabel('X (Bearing Vector Component)')
        ax.set_ylabel('Y (Bearing Vector Component)')
        ax.set_zlabel('Z (Bearing Vector Component)')
        ax.set_title('Catalog (Expected) Bearing Vectors from RA/Dec\n(With Inner Angles)', 
                    fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        ax.legend()
        
        # Add summary text
        if catalog_angles:
            angle_text = ", ".join([f"{angle:.3f}°" for angle in catalog_angles])
            summary_text = f"Catalog Angles: {angle_text}"
        else:
            summary_text = "Catalog Angles: Insufficient vectors for angle calculation"
        
        ax.text2D(0.02, 0.02, summary_text, transform=ax.transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                 fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / 'visualization_4_catalog_bearing_vectors.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
        
        return catalog_bearing_vectors, catalog_angles
        
    def create_coordinate_transformation_flow(self):
        """
        Visualization 5: Complete coordinate transformation pipeline
        Shows the flow from RA/Dec through detector to bearing vectors
        """
        print("\nCreating Visualization 5: Coordinate Transformation Flow...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Complete Coordinate Transformation Pipeline', fontsize=16, fontweight='bold')
        
        # Dynamic color scheme
        num_stars = len(self.scene_data['stars'])
        colors = plt.cm.cool(np.linspace(0, 1, num_stars))
        markers = ['o', 's', '^', 'D', 'P', '*', 'X']

        # 1. Celestial coordinates (RA/Dec)
        ras = [np.degrees(star['ra']) for star in self.scene_data['stars']]
        decs = [np.degrees(star['dec']) for star in self.scene_data['stars']]
        
        for i, (ra, dec) in enumerate(zip(ras, decs)):
            marker = markers[i % len(markers)]
            star_id = self.scene_data['stars'][i]['star_id']
            
            ax1.scatter(ra, dec, color=colors[i], s=200, marker=marker,
                       label=f'Star {star_id}', edgecolors='black', linewidth=1)
            ax1.annotate(f'({ra:.3f}°, {dec:.3f}°)', (ra, dec),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Right Ascension (degrees)')
        ax1.set_ylabel('Declination (degrees)')
        ax1.set_title('1. Celestial Coordinates (RA/Dec)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Detector coordinates (FPA pixels)
        detector_positions = [star['detector_position'] for star in self.scene_data['stars']]
        
        for i, pos in enumerate(detector_positions):
            marker = markers[i % len(markers)]
            star_id = self.scene_data['stars'][i]['star_id']

            ax2.scatter(pos[0], pos[1], color=colors[i], s=200, marker=marker,
                       label=f'Star {star_id}', edgecolors='black', linewidth=1)
            ax2.annotate(f'({pos[0]:.1f}, {pos[1]:.1f})', pos,
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Detector X (pixels)')
        ax2.set_ylabel('Detector Y (pixels)')
        ax2.set_title('2. Focal Plane Array Coordinates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Detected centroids
        # Dynamic color scheme for detected centroids
        num_centroids = len(self.centroids)
        detected_colors = plt.cm.autumn(np.linspace(0, 1, num_centroids))

        for i, centroid in enumerate(self.centroids):
            marker = markers[i % len(markers)]
            
            ax3.scatter(centroid[0], centroid[1], color=detected_colors[i], s=200, marker=marker,
                       label=f'Detected {i+1}', edgecolors='black', linewidth=1)
            ax3.annotate(f'({centroid[0]:.2f}, {centroid[1]:.2f})', centroid,
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Detected X (pixels)')
        ax3.set_ylabel('Detected Y (pixels)')
        ax3.set_title('3. Detected Centroids')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bearing vectors (2D projection)
        for i, bv in enumerate(self.bearing_vectors):
            ax4.arrow(0, 0, bv[0]*1000, bv[1]*1000, color=detected_colors[i],
                     linewidth=3, head_width=15, head_length=15, alpha=0.8, label=f'Detected {i+1}')
            ax4.annotate(f'({bv[0]:.3f}, {bv[1]:.3f})',
                        (bv[0]*1000, bv[1]*1000),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('X Component (×1000)')
        ax4.set_ylabel('Y Component (×1000)')
        ax4.set_title('4. Bearing Vectors (X-Y Projection)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        output_file = self.output_dir / 'visualization_5_coordinate_transformation_flow.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
        
    def create_pipeline_summary_comparison(self):
        """
        Visualization 6: Complete pipeline summary with error analysis
        Shows observed vs catalog comparison and validation metrics
        """
        print("\nCreating Visualization 6: Pipeline Summary & Validation...")
        
        # Calculate catalog bearing vectors for comparison
        catalog_bearing_vectors = []
        for star in self.scene_data['stars']:
            ra, dec = star['ra'], star['dec']
            x, y, z = ra, dec, 1.0
            magnitude = np.sqrt(x*x + y*y + z*z)
            bearing_vector = np.array([x/magnitude, y/magnitude, z/magnitude])
            catalog_bearing_vectors.append(bearing_vector)
        
        # Calculate angles
        observed_angles = []
        catalog_angles = []
        
        # Only calculate angles if we have enough vectors
        min_vectors = min(len(self.bearing_vectors), len(catalog_bearing_vectors))
        if min_vectors >= 2:
            for i in range(min_vectors):
                for j in range(i+1, min_vectors):
                    obs_angle = np.degrees(calculate_vector_angle(self.bearing_vectors[i], self.bearing_vectors[j]))
                    cat_angle = np.degrees(calculate_vector_angle(catalog_bearing_vectors[i], catalog_bearing_vectors[j]))
                    observed_angles.append(obs_angle)
                    catalog_angles.append(cat_angle)
        
        # Calculate errors
        angle_errors = [abs(obs - cat) for obs, cat in zip(observed_angles, catalog_angles)]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Star Pipeline Validation Summary', fontsize=16, fontweight='bold')
        
        # 1. Angle comparison
        if observed_angles and catalog_angles:
            # Create dynamic labels based on actual number of angles
            # Create dynamic labels based on combinations of star pairs
            min_vectors = min(len(self.bearing_vectors), len(catalog_bearing_vectors))
            pair_indices = list(combinations(range(min_vectors), 2))
            x_labels = [f'Pair {i+1}-{j+1}' for i, j in pair_indices]
            
            x_pos = np.arange(len(x_labels))
            
            width = 0.35
            ax1.bar(x_pos - width/2, catalog_angles, width, label='Catalog (Expected)', color='lightblue', alpha=0.7)
            ax1.bar(x_pos + width/2, observed_angles, width, label='Observed (Detected)', color='orange', alpha=0.7)
            
            ax1.set_xlabel('Star Pair')
            ax1.set_ylabel('Angle (degrees)')
            ax1.set_title('Angular Separation Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data for angle comparison', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Angular Separation Comparison - No Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        if observed_angles and catalog_angles:
            for i, (cat, obs) in enumerate(zip(catalog_angles, observed_angles)):
                ax1.text(i - width/2, cat + 0.001, f'{cat:.3f}°', ha='center', va='bottom', fontsize=9)
                ax1.text(i + width/2, obs + 0.001, f'{obs:.3f}°', ha='center', va='bottom', fontsize=9)
        
        # 2. Error analysis
        if angle_errors and 'x_labels' in locals():
            ax2.bar(x_labels, angle_errors, color='red', alpha=0.7)
            ax2.set_xlabel('Star Pair')
            ax2.set_ylabel('Absolute Error (degrees)')
            ax2.set_title('Coordinate Transformation Errors')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No error data available', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Coordinate Transformation Errors - No Data')
        
        # Add value labels
        if angle_errors:
            for i, error in enumerate(angle_errors):
                ax2.text(i, error + max(angle_errors)*0.02, f'{error:.4f}°', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Bearing vector comparison (3D to 2D)
        max_vectors = min(len(self.bearing_vectors), len(catalog_bearing_vectors))
        star_colors = plt.cm.cool(np.linspace(0, 1, max(max_vectors, 1)))
        for i in range(max_vectors):
            obs_bv = self.bearing_vectors[i]
            cat_bv = catalog_bearing_vectors[i]
            color_idx = i % len(star_colors)
            
            # Plot observed
            ax3.arrow(0, 0, obs_bv[0]*1000, obs_bv[1]*1000,
                     color=star_colors[color_idx], linewidth=2, alpha=0.8,
                     head_width=0.3, head_length=0.2, label=f'Observed {i+1}')
            
            # Plot catalog (dashed)
            ax3.arrow(0, 0, cat_bv[0]*1000, cat_bv[1]*1000,
                     color=star_colors[color_idx], linewidth=2, alpha=0.6, linestyle='--',
                     head_width=0.3, head_length=0.2, label=f'Catalog {i+1}')
        
        ax3.set_xlabel('X Component (×1000)')
        ax3.set_ylabel('Y Component (×1000)')
        ax3.set_title('Bearing Vectors: Observed vs Catalog')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. Performance metrics summary
        ax4.axis('off')
        
        # Calculate summary statistics
        if angle_errors:
            max_error = max(angle_errors)
            mean_error = np.mean(angle_errors)
            std_error = np.std(angle_errors)
            status = 'PASSED' if max_error < 0.01 else 'NEEDS REVIEW'
        else:
            max_error = mean_error = std_error = 0.0
            status = 'INSUFFICIENT DATA'
        
        detection_success_rate = len(self.centroids) / len(self.scene_data['stars']) * 100 if self.scene_data['stars'] else 0
        
        # Format angle arrays safely
        catalog_angles_str = ", ".join([f"{angle:.3f}°" for angle in catalog_angles]) if catalog_angles else "N/A"
        observed_angles_str = ", ".join([f"{angle:.3f}°" for angle in observed_angles]) if observed_angles else "N/A"
        angle_errors_str = ", ".join([f"{error:.4f}°" for error in angle_errors]) if angle_errors else "N/A"
        
        summary_text = f"""
        MULTI-STAR PIPELINE VALIDATION RESULTS
        
        Detection Performance:
        • Stars Expected: {len(self.scene_data['stars'])}
        • Stars Detected: {len(self.centroids)}
        • Success Rate: {detection_success_rate:.1f}%
        
        Coordinate Accuracy:
        • Maximum Error: {max_error:.4f}°
        • Mean Error: {mean_error:.4f}°
        • Error Standard Deviation: {std_error:.4f}°
        
        Angular Separations:
        • Expected: [{catalog_angles_str}]
        • Observed: [{observed_angles_str}]
        • Errors: [{angle_errors_str}]
        
        VALIDATION STATUS: {status}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        output_file = self.output_dir / 'visualization_6_pipeline_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
        
    def run_all_visualizations(self):
        """Run all visualizations in sequence"""
        print("Starting Complete Multi-Star Pipeline Visualization Suite...\n")
        
        try:
            # Skip problematic single-star visualization for now
            # self.create_single_star_psf_comparison()
            self.create_multi_star_scene_visualization()
            self.create_observed_bearing_vectors_3d()
            catalog_bvs, catalog_angles = self.create_catalog_bearing_vectors_3d()
            self.create_coordinate_transformation_flow()
            self.create_pipeline_summary_comparison()
            
            print("\nAll visualizations completed successfully!")
            print("\nGenerated files:")
            print(self.output_dir / "visualization_2_multi_star_scene.png")
            print(self.output_dir / "visualization_3_observed_bearing_vectors.png")
            print(self.output_dir / "visualization_4_catalog_bearing_vectors.png")
            print(self.output_dir / "visualization_5_coordinate_transformation_flow.png")
            print(self.output_dir / "visualization_6_pipeline_summary.png")
            print("\nThese visualizations are ready for your team presentation!")
            print("Note: Single-star PSF comparison skipped due to interface compatibility issues.")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Multi-Star Pipeline Visualization Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "catalog_path",
        type=Path,
        help="Path to the catalog CSV file to visualize."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-STAR PIPELINE VISUALIZATION SUITE")
    print("=" * 60)
    print(f"Running visualization for: {args.catalog_path.name}")
    print()
    
    try:
        visualizer = MultiStarVisualizer(args.catalog_path)
        visualizer.run_all_visualizations()
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Set matplotlib style for better plots
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    
    exit_code = main()
    sys.exit(exit_code)