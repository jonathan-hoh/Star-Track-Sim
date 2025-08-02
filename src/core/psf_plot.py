import os
import re
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

# Try to import grid_distortion, but don't fail if it's not available
try:
    from grid_distortion import is_grid_distortion_file, process_grid_distortion_file
    from grid_summary import create_grid_summary_report
    HAS_GRID_DISTORTION = True
except ImportError:
    # Define stub functions to avoid errors when these functions are called
    def is_grid_distortion_file(filepath):
        """Stub function to replace the real is_grid_distortion_file when module is not available"""
        return False
        
    def process_grid_distortion_file(filepath, save_plot=True, output_dir=None, show_plot=True, debug=False):
        """Stub function to replace the real process_grid_distortion_file when module is not available"""
        print(f"Grid distortion processing not available for {filepath}")
        return {}, [], {}
        
    def create_grid_summary_report(results, summary_file):
        """Stub function to replace the real create_grid_summary_report when module is not available"""
        print(f"Grid summary reporting not available")
        
    HAS_GRID_DISTORTION = False

def parse_psf_file(filepath, debug=False):
    """
    Parse a Zemax PSF data file and extract metadata and intensity values.
    
    Args:
        filepath (str): Path to the PSF data file
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (metadata, intensity_data) where metadata is a dictionary and 
               intensity_data is a 2D numpy array
    """
    # Initialize metadata dictionary
    metadata = {
        'title': os.path.basename(filepath),
        'wavelength_range': '',
        'field_angle': 0.0,
        'data_spacing': None,  # Initialize as None, will be float
        'data_area': '',
        'strehl_ratio': None, # Will be float
        'image_grid_size': '', # Original string format
        'image_grid_dim': None, # Tuple (width, height)
        'pupil_grid_size': '', # Original string format
        'pupil_grid_dim': None,  # Tuple (width, height)
        'center_point': '',
        'center_coordinates': '',
        'centroid_offset': '',
        'centroid_coordinates': ''
    }
    
    # Initialize content as empty list
    content = []
    
    try:
        # Try to open the file with UTF-16 encoding first
        try:
            with open(filepath, 'r', encoding='utf-16') as file:
                content = file.readlines()
                if debug:
                    print("Successfully opened with UTF-16 encoding")
        except UnicodeDecodeError:
            # If UTF-16 fails, try UTF-8 with BOM handling
            try:
                with open(filepath, 'r', encoding='utf-8-sig') as file:
                    content = file.readlines()
                    if debug:
                        print("Successfully opened with UTF-8-sig encoding")
            except UnicodeDecodeError:
                # If UTF-8-sig fails, try regular UTF-8
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.readlines()
                        if debug:
                            print("Successfully opened with UTF-8 encoding")
                except UnicodeDecodeError:
                    # Last resort: use latin-1 which can read any byte sequence
                    with open(filepath, 'r', encoding='latin-1') as file:
                        content = file.readlines()
                        if debug:
                            print("Successfully opened with latin-1 encoding")
        
        if debug:
            print(f"File contains {len(content)} lines")
        
        # Extract metadata from header
        data_start_idx = 0
        values_line_found = False
        
        for i, line in enumerate(content):
            line_stripped = line.strip()
            if not line_stripped:  # Skip empty lines
                continue
                
            if debug and i < 30:  # Print first few lines for debugging
                print(f"Line {i}: {line_stripped}")
                
            # Look for field angle
            if 'to' in line_stripped and 'µm at' in line_stripped:
                match = re.search(r'([\d\.]+) to ([\d\.]+) µm at ([\d\.]+) \(deg\)', line_stripped)
                if match:
                    metadata['wavelength_range'] = f"{match.group(1)} to {match.group(2)} µm"
                    metadata['field_angle'] = float(match.group(3))
                    if debug:
                        print(f"Found field angle: {metadata['field_angle']}")
            
            # Data spacing
            elif 'data spacing' in line_stripped.lower(): # Broader initial check
                if debug:
                    print(f"DEBUG parse_psf_file: Attempting to match 'Data spacing' in line: '{line_stripped}'")
                # Regex made more flexible: allows for optional 'is', optional ':', flexible spacing
                match = re.search(r'Data\s+spacing(?:\s*is)?\s*:?\s*([\d\.]+)\s*µm', line_stripped, re.IGNORECASE)
                if match:
                    try:
                        metadata['data_spacing'] = float(match.group(1))
                        if debug:
                            print(f"DEBUG parse_psf_file: Successfully parsed data spacing: {metadata['data_spacing']} µm from line: '{line_stripped}'")
                    except ValueError:
                        if debug:
                            print(f"DEBUG parse_psf_file: Could not convert data spacing '{match.group(1)}' to float from line: '{line_stripped}'")
                elif debug: 
                    print(f"DEBUG parse_psf_file: Regex for 'Data spacing' did NOT match line: '{line_stripped}'")
            
            # Data area
            elif 'Data area is' in line_stripped:
                match = re.search(r'Data area is ([\d\.]+) by ([\d\.]+) µm', line_stripped)
                if match:
                    metadata['data_area'] = f"{match.group(1)} x {match.group(2)} µm"
                    if debug:
                        print(f"Found data area: {metadata['data_area']}")
            
            # Strehl ratio
            elif 'Strehl ratio:' in line_stripped:
                match = re.search(r'Strehl ratio: ([\\d\\.]+)', line_stripped)
                if match:
                    try:
                        metadata['strehl_ratio'] = float(match.group(1))
                        if debug:
                            print(f"Found Strehl ratio: {metadata['strehl_ratio']}")
                    except ValueError:
                        if debug:
                            print(f"Could not convert Strehl ratio '{match.group(1)}' to float")
            
            # Image grid size
            elif 'image grid size' in line_stripped.lower(): 
                if debug:
                    print(f"DEBUG parse_psf_file: Attempting to match 'Image grid size' in line: '{line_stripped}'")
                # Regex made more flexible: optional ':', flexible spacing
                match = re.search(r'Image\s+grid\s+size\s*:?\s*([\d]+)\s+by\s+([\d]+)', line_stripped, re.IGNORECASE)
                if match:
                    metadata['image_grid_size'] = f"{match.group(1)} x {match.group(2)}"
                    try:
                        metadata['image_grid_dim'] = (int(match.group(1)), int(match.group(2))) 
                        if debug:
                            print(f"DEBUG parse_psf_file: Found image grid size: {metadata['image_grid_size']} -> dim: {metadata['image_grid_dim']} from line: '{line_stripped}'")
                    except ValueError:
                        if debug:
                            print(f"DEBUG parse_psf_file: Could not convert image grid dims '{match.group(1)}', '{match.group(2)}' to int from: '{line_stripped}'")
                elif debug:
                     print(f"DEBUG parse_psf_file: Regex for 'Image grid size' did NOT match line: '{line_stripped}'")
            
            # Pupil grid size
            elif 'pupil grid size' in line_stripped.lower(): 
                if debug:
                    print(f"DEBUG parse_psf_file: Attempting to match 'Pupil grid size' in line: '{line_stripped}'")
                # Regex made more flexible: optional ':', flexible spacing
                match = re.search(r'Pupil\s+grid\s+size\s*:?\s*([\d]+)\s+by\s+([\d]+)', line_stripped, re.IGNORECASE)
                if match:
                    metadata['pupil_grid_size'] = f"{match.group(1)} x {match.group(2)}"
                    try:
                        metadata['pupil_grid_dim'] = (int(match.group(1)), int(match.group(2)))
                        if debug:
                            print(f"Found pupil grid size: {metadata['pupil_grid_size']} -> dim: {metadata['pupil_grid_dim']}")
                    except ValueError:
                        if debug:
                            print(f"Could not convert pupil grid dims '{match.group(1)}', '{match.group(2)}' to int")
            
            # Center point
            elif 'Center point is:' in line_stripped:
                match = re.search(r'Center point is: *(\d+), *(\d+)', line_stripped)
                if match:
                    metadata['center_point'] = (int(match.group(1)), int(match.group(2)))
                    if debug:
                        print(f"Found center point: {metadata['center_point']}")
            
            # Find where values start
            elif 'Values are relative intensity' in line_stripped or 'Values are in relative intensity' in line_stripped:
                values_line_found = True
                data_start_idx = i + 1
                if debug:
                    print(f"Found data start at line {data_start_idx}")
                break
        
        # If we didn't find the values line, look for scientific notation
        if not values_line_found:
            if debug:
                print("Values line not found, looking for scientific notation...")
            
            for i, line in enumerate(content):
                line_stripped = line.strip()
                if re.search(r'\d+\.\d+E[-+]\d+', line_stripped):
                    data_start_idx = i
                    values_line_found = True
                    if debug:
                        print(f"Found data start at line {data_start_idx} with scientific notation")
                    break
        
        # Extract field angle from filename if not found in content
        if metadata['field_angle'] == 0.0:
            filename = os.path.basename(filepath)
            match = re.search(r'(\d+(?:\.\d+)?)deg', filename)
            if match:
                metadata['field_angle'] = float(match.group(1))
                if debug:
                    print(f"Extracted field angle from filename: {metadata['field_angle']}")
        
        # Extract intensity data
        intensity_data = []
        
        for i in range(data_start_idx, len(content)):
            line = content[i].strip()
            if not line:  # Skip empty lines
                continue
            
            # Look for scientific notation values
            sci_values = re.findall(r'(\d+\.\d+E[-+]\d+)', line)
            if sci_values:
                try:
                    row_data = [float(val) for val in sci_values]
                    intensity_data.append(row_data)
                    if debug and len(intensity_data) <= 5:
                        print(f"Found data row {len(intensity_data)}: {len(row_data)} values")
                except ValueError as e:
                    if debug:
                        print(f"Error converting values: {str(e)}")
            else:
                # Try space-separated values as fallback
                try:
                    row_values = [float(val) for val in line.split()]
                    if row_values:
                        intensity_data.append(row_values)
                        if debug and len(intensity_data) <= 5:
                            print(f"Found data row {len(intensity_data)}: {len(row_values)} values")
                except ValueError:
                    if debug:
                        print(f"Non-numeric data at line {i}: {line[:30]}...")
                    continue
        
        # Convert to numpy array
        if intensity_data:
            # Check if all rows have the same length
            row_lengths = [len(row) for row in intensity_data]
            if row_lengths:
                max_length = max(row_lengths)
                
                if not all(len(row) == max_length for row in intensity_data):
                    if debug:
                        print(f"Padding rows to uniform length of {max_length}")
                    padded_data = []
                    for row in intensity_data:
                        if len(row) < max_length:
                            padded_row = np.pad(row, (0, max_length - len(row)), 'constant')
                            padded_data.append(padded_row)
                        else:
                            padded_data.append(row)
                    intensity_data = np.array(padded_data)
                else:
                    intensity_data = np.array(intensity_data)
                
                if debug:
                    print(f"Final data shape: {intensity_data.shape}")
                    if intensity_data.size > 0:
                        print(f"Data range: {np.min(intensity_data)} to {np.max(intensity_data)}")
            else:
                intensity_data = np.array([])
                if debug:
                    print("No intensity data extracted")
        else:
            intensity_data = np.array([])
            if debug:
                print("No intensity data extracted from file")
        
        return metadata, intensity_data
    
    except Exception as e:
        print(f"Error parsing file {filepath}: {str(e)}")
        # Try to debug encoding issues
        try:
            with open(filepath, 'rb') as f:
                first_bytes = f.read(4)
                print(f"First bytes: {' '.join(f'{b:02x}' for b in first_bytes)}")
                if first_bytes.startswith(b'\xff\xfe'):
                    print("File appears to be UTF-16 LE encoded")
        except Exception as e2:
            print(f"Could not examine file bytes: {str(e2)}")
        return {}, np.array([])

def analyze_psf(metadata, intensity_data):
    """
    Analyze PSF data to extract key metrics
    
    Args:
        metadata (dict): Metadata dictionary
        intensity_data (array): 2D array of intensity values
        
    Returns:
        dict: Dictionary of analysis results
    """
    analysis = {
        'peak_intensity': 0,
        'peak_position': (0, 0),
        'fwhm_x': 0,
        'fwhm_y': 0,
        'encircled_energy_80pct': 0,
        'errors': []
    }
    
    # Check if we have valid data
    if intensity_data.size == 0:
        analysis['errors'].append("No intensity data found")
        return analysis
    
    try:
        # Find peak intensity and position
        peak_intensity = np.max(intensity_data)
        peak_idx = np.unravel_index(np.argmax(intensity_data), intensity_data.shape)
        analysis['peak_intensity'] = float(peak_intensity)
        analysis['peak_position'] = peak_idx
        
        # Calculate center of mass
        com = center_of_mass(intensity_data)
        analysis['center_of_mass'] = tuple(float(val) for val in com)
        
        # Calculate FWHM in X and Y directions
        half_max = peak_intensity / 2
        
        # FWHM in X direction (through peak)
        row = intensity_data[peak_idx[0], :]
        try:
            # Try to fit a Gaussian to improve FWHM estimate
            x = np.arange(len(row))
            
            # Define Gaussian function
            def gaussian(x, a, x0, sigma):
                return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
            
            # Initial guess
            p0 = [peak_intensity, peak_idx[1], 1]
            
            # Try curve fitting
            try:
                popt, _ = curve_fit(gaussian, x, row, p0=p0)
                sigma = abs(popt[2])
                analysis['fwhm_x'] = 2.355 * sigma * metadata.get('data_spacing', 1.0)  # Convert to microns
            except:
                # Fallback to simple threshold method
                above_half = row >= half_max
                if np.any(above_half):
                    left_idx = np.min(np.where(above_half)[0])
                    right_idx = np.max(np.where(above_half)[0])
                    analysis['fwhm_x'] = (right_idx - left_idx) * metadata.get('data_spacing', 1.0)
        except Exception as e:
            analysis['errors'].append(f"Error calculating FWHM in X: {str(e)}")
        
        # FWHM in Y direction (through peak)
        col = intensity_data[:, peak_idx[1]]
        try:
            # Try to fit a Gaussian
            y = np.arange(len(col))
            
            # Try curve fitting
            try:
                p0 = [peak_intensity, peak_idx[0], 1]
                popt, _ = curve_fit(gaussian, y, col, p0=p0)
                sigma = abs(popt[2])
                analysis['fwhm_y'] = 2.355 * sigma * metadata.get('data_spacing', 1.0)
            except:
                # Fallback to simple threshold method
                above_half = col >= half_max
                if np.any(above_half):
                    top_idx = np.min(np.where(above_half)[0])
                    bottom_idx = np.max(np.where(above_half)[0])
                    analysis['fwhm_y'] = (bottom_idx - top_idx) * metadata.get('data_spacing', 1.0)
        except Exception as e:
            analysis['errors'].append(f"Error calculating FWHM in Y: {str(e)}")
        
        # Calculate radius containing 80% of energy
        try:
            # Create distance map from peak
            y, x = np.indices(intensity_data.shape)
            r = np.sqrt((x - peak_idx[1])**2 + (y - peak_idx[0])**2)
            
            # Sort pixels by distance from peak
            sorted_idx = np.argsort(r.flatten())
            sorted_r = r.flatten()[sorted_idx]
            sorted_intensity = intensity_data.flatten()[sorted_idx]
            
            # Calculate cumulative energy
            total_energy = np.sum(intensity_data)
            cumulative_energy = np.cumsum(sorted_intensity) / total_energy
            
            # Find radius containing 80% energy
            idx_80pct = np.searchsorted(cumulative_energy, 0.8)
            if idx_80pct < len(sorted_r):
                analysis['encircled_energy_80pct'] = sorted_r[idx_80pct] * metadata.get('data_spacing', 1.0)
        except Exception as e:
            analysis['errors'].append(f"Error calculating encircled energy: {str(e)}")
    
    except Exception as e:
        analysis['errors'].append(f"Error in PSF analysis: {str(e)}")
    
    return analysis


def visualize_psf(metadata, intensity_data, analysis_results=None, log_scale=True, save_path=None, show_plot=True, show_grid=True, pixel_size=None):
    """
    Create a visualization of the PSF data
    
    Args:
        metadata (dict): Metadata dictionary
        intensity_data (array): 2D array of intensity values
        analysis_results (dict, optional): Results from analyze_psf
        log_scale (bool): Use logarithmic color scale
        save_path (str, optional): Path to save the figure
        show_plot (bool): Whether to display the plot
        show_grid (bool): Whether to show pixel grid overlay
        pixel_size (float): Size of detector pixels in microns. If None, no grid is shown.
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if intensity_data.size == 0:
        print("No intensity data to visualize")
        return None
    
    # Get dimensions for proper scaling
    data_spacing = metadata.get('data_spacing', 1.0)
    ny, nx = intensity_data.shape
    extent = [-nx/2 * data_spacing, nx/2 * data_spacing, 
              -ny/2 * data_spacing, ny/2 * data_spacing]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create image
    if log_scale:
        # Add small value to avoid log(0)
        vmin = np.min(intensity_data[intensity_data > 0]) / 10 if np.any(intensity_data > 0) else 1e-10
        norm = LogNorm(vmin=vmin, vmax=np.max(intensity_data))
        im = ax.imshow(intensity_data, extent=extent, origin='lower', 
                      cmap='viridis', norm=norm)
        plt.colorbar(im, ax=ax, label='Log Intensity (relative)')
    else:
        im = ax.imshow(intensity_data, extent=extent, origin='lower', 
                      cmap='viridis')
        plt.colorbar(im, ax=ax, label='Intensity (relative)')

    # Add pixel grid overlay if requested
    if show_grid and pixel_size is not None:
        # Calculate grid lines
        x_min, x_max = extent[0], extent[1]
        y_min, y_max = extent[2], extent[3]

        # Generate grid lines at pixel_size intervals
        x_lines = np.arange(np.floor(x_min/pixel_size) * pixel_size,
                           np.ceil(x_max/pixel_size) * pixel_size + pixel_size,
                           pixel_size)
        y_lines = np.arange(np.floor(y_min/pixel_size) * pixel_size,
                           np.ceil(y_max/pixel_size) * pixel_size + pixel_size,
                           pixel_size)

        # Plot grid lines
        for x in x_lines:
            ax.axvline(x=x, color='white', linestyle=':', alpha=0.3, linewidth=0.5)
        for y in y_lines:
            ax.axhline(y=y, color='white', linestyle=':', alpha=0.3, linewidth=0.5)

        # Add grid size to plot title
        grid_text = f" | Pixel Size: {pixel_size} µm"
    else:
        grid_text = ""

    # Add metadata as text box
    field_angle = metadata.get('field_angle', 'N/A')
    wavelength = metadata.get('wavelength_range', 'N/A')
    strehl = metadata.get('strehl_ratio', 'N/A')

    title = f"PSF at Field Angle: {field_angle}° | Wavelength: {wavelength} | Strehl Ratio: {strehl}{grid_text}"
    ax.set_title(title)

    # Add analysis results if available
    if analysis_results:
        info_text = f"Peak Intensity: {analysis_results.get('peak_intensity', 'N/A'):.4f}\n"
        if 'fwhm_x' in analysis_results and 'fwhm_y' in analysis_results:
            info_text += f"FWHM: {analysis_results['fwhm_x']:.2f} × {analysis_results['fwhm_y']:.2f} µm\n"
        if 'encircled_energy_80pct' in analysis_results:
            info_text += f"80% Energy Radius: {analysis_results['encircled_energy_80pct']:.2f} µm"
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.03, 0.97, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    # Set labels
    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')

    # Save if requested
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")

    # Show if requested
    if show_plot:
        plt.tight_layout()
        plt.show()  # Ensure this is called to display the plot
    else:
        plt.close(fig)  # Only close the plot if show_plot is False
    
    return fig

def process_psf_file(filepath, save_plot=True, output_dir=None, log_scale=True, show_plot=True, debug=False, show_grid=True, pixel_size=None):
    """
    Process a single PSF file: parse, analyze, and visualize

    Args:
        filepath (str): Path to the PSF file
        save_plot (bool): Whether to save the plot
        output_dir (str, optional): Directory to save output files
        log_scale (bool): Use logarithmic color scale
        show_plot (bool): Whether to display the plot
        debug (bool): Whether to print detailed debug information
        show_grid (bool): Whether to show pixel grid overlay
        pixel_size (float): Size of detector pixels in microns

    Returns:
        tuple: (metadata, intensity_data, analysis_results)
    """
    if debug:
        print(f"Debug: pixel_size = {pixel_size}")

    print(f"Processing file: {filepath}")
    
    # Parse file
    if debug:
        print(f"\nDEBUG: Parsing file {filepath}")
    
    metadata, intensity_data = parse_psf_file(filepath, debug=debug)
    
    if not metadata or intensity_data.size == 0:
        print(f"Failed to extract valid data from {filepath}")
        if debug:
            # Try to give more information about the file
            try:
                with open(filepath, 'r') as file:
                    head = ''.join(file.readlines(20))  # Read first 20 lines
                    print(f"\nFirst few lines of the file:\n{head}")
            except Exception as e:
                print(f"Could not read file for debugging: {str(e)}")
        return {}, np.array([]), {}
    
    if debug:
        print(f"\nExtracted metadata: {metadata}")
        print(f"Intensity data shape: {intensity_data.shape}")
        if intensity_data.size > 0:
            print(f"Intensity range: {np.min(intensity_data)} to {np.max(intensity_data)}")
    
    # Analyze PSF
    analysis_results = analyze_psf(metadata, intensity_data)
    
    if debug:
        print(f"\nAnalysis results: {analysis_results}")
    
    # Prepare output path
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(filepath)
        
        # Create output directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create output filename
        filename = os.path.basename(filepath)
        base, _ = os.path.splitext(filename)
        plot_path = os.path.join(output_dir, f"{base}_psf_plot.png")
    else:
        plot_path = None
    
    # Visualize
    try:
        visualize_psf(metadata, intensity_data, analysis_results, 
                     log_scale=log_scale, save_path=plot_path, show_plot=show_plot,
                     show_grid=show_grid, pixel_size=pixel_size)
    except Exception as e:
        print(f"Error visualizing PSF: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\nSummary for {os.path.basename(filepath)}:")
    print(f"  Field Angle: {metadata.get('field_angle', 'N/A')}°")
    print(f"  Strehl Ratio: {metadata.get('strehl_ratio', 'N/A')}")
    
    if intensity_data.size > 0:
        print(f"  Data Shape: {intensity_data.shape}")
    
    if 'fwhm_x' in analysis_results and 'fwhm_y' in analysis_results:
        print(f"  FWHM: {analysis_results['fwhm_x']:.2f} × {analysis_results['fwhm_y']:.2f} µm")
    
    if 'encircled_energy_80pct' in analysis_results:
        print(f"  80% Energy Radius: {analysis_results['encircled_energy_80pct']:.2f} µm")
    
    if analysis_results.get('errors'):
        print("  Warnings/Errors:")
        for error in analysis_results['errors']:
            print(f"    - {error}")
    
    return metadata, intensity_data, analysis_results

def batch_process_psf_files(directory, pattern="*_deg.txt", 
                          save_plot=True, output_dir=None, log_scale=True,
                          show_plot=False, summary_file=None, debug=False):
    """
    Process multiple PSF files in a directory
    
    Args:
        directory (str): Directory containing PSF files
        pattern (str): Glob pattern to match PSF files
        save_plot (bool): Whether to save plots
        output_dir (str, optional): Directory to save output files
        log_scale (bool): Use logarithmic color scale
        show_plot (bool): Whether to display plots
        summary_file (str, optional): Path to save summary report
        debug (bool): Whether to print debugging information
        
    Returns:
        dict: Dictionary mapping filenames to analysis results
    """
    # Find all matching files
    if os.path.isfile(directory):
        # If directory is actually a file, just process that file
        files = [directory]
    else:
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found matching pattern '{pattern}' in {directory}")
        if debug:
            # List all files in the directory to help diagnose pattern issues
            try:
                all_files = os.listdir(directory)
                print(f"Files in directory: {all_files}")
            except Exception as e:
                print(f"Error listing directory contents: {str(e)}")
        return {}
    
    print(f"Found {len(files)} files to process:")
    if debug:
        for f in files:
            print(f"  - {os.path.basename(f)}")
    
    # Process each file
    results = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        metadata, intensity_data, analysis = process_psf_file(
            file_path, save_plot=save_plot, output_dir=output_dir,
            log_scale=log_scale, show_plot=show_plot, debug=debug
        )
        
        if metadata and intensity_data.size > 0:
            results[filename] = {
                'metadata': metadata,
                'analysis': analysis
            }
    
    # Create summary report if requested
    if summary_file and results:
        create_summary_report(results, summary_file)
    
    return results

def create_summary_report(results, output_path):
    """
    Create a summary report of PSF analysis results
    
    Args:
        results (dict): Dictionary mapping filenames to analysis results
        output_path (str): Path to save the report
    """
    try:
        with open(output_path, 'w') as f:
            f.write("# PSF Analysis Summary Report\n\n")
            
            # Create comparison table
            f.write("## Comparison Table\n\n")
            f.write("| File | Field Angle (°) | Strehl Ratio | FWHM X (µm) | FWHM Y (µm) | 80% Energy Radius (µm) |\n")
            f.write("|------|----------------|--------------|-------------|-------------|------------------------|\n")
            
            for filename, data in results.items():
                metadata = data['metadata']
                analysis = data['analysis']
                
                field_angle = metadata.get('field_angle', 'N/A')
                strehl = metadata.get('strehl_ratio', 'N/A')
                fwhm_x = analysis.get('fwhm_x', 'N/A')
                fwhm_y = analysis.get('fwhm_y', 'N/A')
                ee80 = analysis.get('encircled_energy_80pct', 'N/A')
                
                # Format numeric values
                if isinstance(fwhm_x, (int, float)):
                    fwhm_x = f"{fwhm_x:.2f}"
                if isinstance(fwhm_y, (int, float)):
                    fwhm_y = f"{fwhm_y:.2f}"
                if isinstance(ee80, (int, float)):
                    ee80 = f"{ee80:.2f}"
                
                f.write(f"| {filename} | {field_angle} | {strehl} | {fwhm_x} | {fwhm_y} | {ee80} |\n")
            
            # Individual file details
            f.write("\n## Individual File Details\n\n")
            
            for filename, data in results.items():
                metadata = data['metadata']
                analysis = data['analysis']
                
                f.write(f"### {filename}\n\n")
                
                # Metadata
                f.write("#### Metadata\n\n")
                for key, value in metadata.items():
                    if key not in ['title', 'center_coordinates', 'centroid_offset', 'centroid_coordinates']:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                
                # Analysis results
                f.write("\n#### Analysis\n\n")
                for key, value in analysis.items():
                    if key != 'errors' and not key.startswith('_'):
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                
                # Errors/warnings
                if analysis.get('errors'):
                    f.write("\n#### Warnings/Errors\n\n")
                    for error in analysis['errors']:
                        f.write(f"- {error}\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {output_path}")
    
    except Exception as e:
        print(f"Error creating summary report: {str(e)}")

# PSF Generation Label Mapping
PSF_GENERATION_LABELS = {
    4: "Final Optic Design - Cold",
    5: "Final Optic Design - 20°C", 
    6: "Final Optic Design - Hot"
}

def get_psf_generation_label(gen_num, use_descriptive_labels=True):
    """
    Get a descriptive label for a PSF generation number.
    
    Args:
        gen_num (int): PSF generation number
        use_descriptive_labels (bool): If True, use descriptive labels for supported generations,
                                     otherwise use default "Generation X" format
    
    Returns:
        str: Descriptive label for the generation
    """
    if use_descriptive_labels and gen_num in PSF_GENERATION_LABELS:
        return PSF_GENERATION_LABELS[gen_num]
    else:
        return f"Generation {gen_num}"

def discover_psf_generations_and_angles(base_psf_dir="PSF_sims"):
    """
    Discovers PSF generations and available field angles within a base directory.

    Args:
        base_psf_dir (str): The root directory containing 'Gen_X' subfolders.

    Returns:
        dict: A dictionary where keys are generation numbers (int) and values are
              sorted lists of available field angles (float).
              Returns an empty dict if base_psf_dir doesn't exist or no valid
              generations/angles are found.
    """
    if not os.path.isdir(base_psf_dir):
        print(f"Error: Base PSF directory '{base_psf_dir}' not found.")
        return {}

    discovered_data = {}
    gen_pattern = re.compile(r"Gen_(\d+)")
    angle_file_pattern = re.compile(r"([\d\.]+)_deg\.txt")

    for item in os.listdir(base_psf_dir):
        gen_path = os.path.join(base_psf_dir, item)
        if os.path.isdir(gen_path):
            match = gen_pattern.fullmatch(item) # Use fullmatch for 'Gen_X' exactly
            if match:
                try:
                    gen_num = int(match.group(1))
                    angles_in_gen = []
                    if not os.access(gen_path, os.R_OK):
                        print(f"Warning: Cannot read directory {gen_path}. Skipping.")
                        continue

                    for psf_filename in os.listdir(gen_path):
                        if os.path.isfile(os.path.join(gen_path, psf_filename)):
                            angle_match = angle_file_pattern.fullmatch(psf_filename)
                            if angle_match:
                                try:
                                    angle = float(angle_match.group(1))
                                    angles_in_gen.append(angle)
                                except ValueError:
                                    print(f"Warning: Could not parse angle from filename '{psf_filename}' in {gen_path}")
                    
                    if angles_in_gen:
                        discovered_data[gen_num] = sorted(list(set(angles_in_gen))) # Ensure unique and sorted
                    else:
                        print(f"Warning: No valid PSF files (e.g., 'X.Y_deg.txt') found in generation directory: {gen_path}")
                except ValueError:
                    print(f"Warning: Could not parse generation number from directory name '{item}'. Skipping.")
            # else: (Optional: print warning if a dir doesn't match Gen_X format but is in base_psf_dir)
            #    print(f"Info: Directory '{item}' does not match 'Gen_X' format. Skipping.")

    if not discovered_data:
        print(f"Warning: No valid PSF generations or angle files found in '{base_psf_dir}'.")
        
    return discovered_data

def main():
    """Main function to handle command line interface"""
    parser = argparse.ArgumentParser(description="Process Zemax data files (PSF and Grid Distortion)")

    # Define file input arguments - allow for single file or directory
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("path", nargs="?", help="Path to data file or directory")

    # Optional arguments
    parser.add_argument("--pattern", default="*_deg.txt",
                      help="File pattern to match (e.g., '*_deg.txt')")
    parser.add_argument("--output", dest="output_dir",
                      help="Directory to save output files")
    parser.add_argument("--no-save", dest="save_plot", action="store_false",
                      help="Don't save plots")
    parser.add_argument("--linear", dest="log_scale", action="store_false",
                      help="Use linear color scale instead of logarithmic (PSF only)")
    parser.add_argument("--no-show", dest="show_plot", action="store_false",
                      help="Don't display plots")
    parser.add_argument("--summary", dest="summary_file",
                      help="Path to save summary report")
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed output")
    parser.add_argument("--debug", action="store_true",
                      help="Print debugging information")
    parser.add_argument("--psf-only", action="store_true",
                      help="Only process PSF files")
    parser.add_argument("--grid-only", action="store_true",
                      help="Only process Grid Distortion files")
    parser.add_argument("--no-grid", dest="show_grid", action="store_false",
                      help="Don't show pixel grid overlay")
    parser.add_argument("--pixel-size", type=float,
                      help="Detector pixel size in microns")

    args = parser.parse_args()

    # Check if the path exists
    if not os.path.exists(args.path):
        print(f"Error: The path '{args.path}' does not exist")
        sys.exit(1)

    # Process files
    if os.path.isfile(args.path):
        # Single file mode
        if args.grid_only or (not args.psf_only and HAS_GRID_DISTORTION and is_grid_distortion_file(args.path)):
            # Process as Grid Distortion file (if grid_distortion module is available)
            if not HAS_GRID_DISTORTION:
                print(f"Warning: Grid distortion module not available. Cannot process {args.path} as grid distortion file.")
                if not args.psf_only:
                    print("Trying to process as PSF file instead...")
                    metadata, intensity_data, analysis_results = process_psf_file(
                        args.path,
                        save_plot=args.save_plot,
                        output_dir=args.output_dir,
                        log_scale=args.log_scale,
                        show_plot=args.show_plot,
                        debug=args.debug,
                        show_grid=args.show_grid,
                        pixel_size=args.pixel_size
                    )
            else:
                process_grid_distortion_file(
                    args.path,
                    save_plot=args.save_plot,
                    output_dir=args.output_dir,
                    show_plot=args.show_plot,
                    debug=args.debug
                )
        elif args.psf_only or not args.grid_only:
            # Process as PSF file
            metadata, intensity_data, analysis_results = process_psf_file(
                args.path,
                save_plot=args.save_plot,
                output_dir=args.output_dir,
                log_scale=args.log_scale,
                show_plot=args.show_plot,
                debug=args.debug,
                show_grid=args.show_grid,
                pixel_size=args.pixel_size
            )
    else:
        # Directory mode
        search_pattern = os.path.join(args.path, args.pattern)
        files = glob.glob(search_pattern)

        if not files:
            print(f"No files found matching pattern '{args.pattern}' in {args.path}")
            sys.exit(1)

        print(f"Found {len(files)} files to process")

        psf_results = {}
        grid_results = {}

        for filepath in files:
            if args.grid_only or (not args.psf_only and HAS_GRID_DISTORTION and is_grid_distortion_file(filepath)):
                # Process as Grid Distortion file (if grid_distortion module is available)
                if not HAS_GRID_DISTORTION:
                    print(f"Warning: Grid distortion module not available. Cannot process {filepath} as grid distortion file.")
                    if not args.psf_only:
                        print("Trying to process as PSF file instead...")
                        metadata, intensity_data, analysis = process_psf_file(
                            filepath,
                            save_plot=args.save_plot,
                            output_dir=args.output_dir,
                            log_scale=args.log_scale,
                            show_plot=args.show_plot,
                            debug=args.debug,
                            show_grid=args.show_grid,
                            pixel_size=args.pixel_size
                        )
                        
                        if metadata and intensity_data.size > 0:
                            psf_results[os.path.basename(filepath)] = {
                                'metadata': metadata,
                                'analysis': analysis
                            }
                else:
                    metadata, grid_data, analysis = process_grid_distortion_file(
                        filepath,
                        save_plot=args.save_plot,
                        output_dir=args.output_dir,
                        show_plot=args.show_plot,
                        debug=args.debug
                    )

                    if metadata and len(grid_data) > 0:
                        grid_results[os.path.basename(filepath)] = {
                            'metadata': metadata,
                            'analysis': analysis
                        }
            elif args.psf_only or not args.grid_only:
                # Process as PSF file
                metadata, intensity_data, analysis = process_psf_file(
                    filepath,
                    save_plot=args.save_plot,
                    output_dir=args.output_dir,
                    log_scale=args.log_scale,
                    show_plot=args.show_plot,
                    debug=args.debug,
                    show_grid=args.show_grid,
                    pixel_size=args.pixel_size
                )

                if metadata and intensity_data.size > 0:
                    psf_results[os.path.basename(filepath)] = {
                        'metadata': metadata,
                        'analysis': analysis
                    }

        # Create summary reports if requested
        if args.summary_file:
            if psf_results:
                psf_summary_file = args.summary_file
                if args.grid_only and grid_results:
                    # Add _psf suffix if we're also processing grid files
                    base, ext = os.path.splitext(args.summary_file)
                    psf_summary_file = f"{base}_psf{ext}"
                create_summary_report(psf_results, psf_summary_file)

            if grid_results and HAS_GRID_DISTORTION:
                grid_summary_file = args.summary_file
                if args.psf_only and psf_results:
                    # Add _grid suffix if we're also processing psf files
                    base, ext = os.path.splitext(args.summary_file)
                    grid_summary_file = f"{base}_grid{ext}"
                create_grid_summary_report(grid_results, grid_summary_file)

if __name__ == "__main__":
    main()