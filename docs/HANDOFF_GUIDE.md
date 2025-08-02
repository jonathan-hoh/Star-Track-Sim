# Star Tracker Radiometry Pipeline - Developer Handoff Guide

**Version**: Complete Radiometry Simulation System  (With FPA Transformation)
**Date**: June 2025  
**Author**: JR
**Purpose**: Developer knowledge transfer and codebase maintenance guide

---

## Executive Summary for New Developers

Welcome soldier. You're inheriting a star tracker simulation pipeline that models the complete signal chain from optical PSFs to bearing vector calculations. 

**Core Architecture**: Complete radiometric simulation with dual analysis capabilities - high-resolution PSF analysis for optical design work and detector-projected analysis for realistic hardware performance prediction.

This guide has been created to help maintain, debug, and extend this system while preserving its validated performance and ensuring continued reliability of the complete simulation chain.

---

## System Architecture Overview

### Complete Radiometric Chain

```
Stellar Magnitude → Photon Flux → Optical System → PSF → Detector Response → 
Digital Counts → Thresholding → Connected Components → Centroiding → Bearing Vectors
```

Each step is modeled with realistic physics and validated against expected performance metrics.

### Core Components

**1. Radiometric Modeling** (`starcamera_model.py`)
- Stellar photometry calculations
- Optical system transmission 
- Detector quantum efficiency and noise
- Temperature-dependent effects

**2. PSF Processing** (`psf_plot.py`, `psf_photon_simulation.py`)
- Zemax PSF file parsing with robust metadata handling
- Photon-based Monte Carlo simulation with Poisson noise
- Multiple analysis scales: optical resolution and detector resolution

**3. Detection and Centroiding** (`identify.py`, `star_tracker_pipeline.py`)
- Adaptive local thresholding  
- Connected components analysis
- Brightness-based region selection
- Sub-pixel moment-based centroiding

**4. Analysis Framework** (`star_tracker_pipeline.py`)
- Dual analysis modes for comprehensive validation
- Monte Carlo statistical evaluation
- Parameter space exploration
- Performance metric calculation

### Key Design Philosophy

**Physics-Based Validation**: Every algorithm decision traces back to physical principles. The system works because it models reality correctly.

**Dual Analysis Pattern**: Most functions provide both high-resolution PSF analysis and detector-projected analysis, enabling validation and comparison.

**Multi-Generation PSF Handling**: The pipeline is designed to handle multiple "Generations" of PSF data, each with potentially different simulation parameters (grid size, pixel spacing). The system automatically discovers and adapts to these different datasets.

**Conservative Defaults**: System defaults to safe, well-tested behavior. Advanced features are opt-in.

---

## Development Environment Setup

### Prerequisites and Dependencies

```bash
# Core scientific stack
pip install numpy scipy matplotlib pandas

# Computer vision (for connected components)
pip install opencv-python

# While not required, it is recommended to install jupyter notebook for visualization purposes
pip install pytest black flake8 ipython jupyter
```

**Critical OpenCV Note**: The codebase uses `cv2.connectedComponentsWithStats()` extensively. Linters show "member not found" warnings, but **this is cosmetic** - the functionality works correctly. Do not "fix" these warnings.

### Development Workflow

```bash
# 1. Validate complete installation and interactive prompts
python debug_centroiding.py
# (Select Gen 2, then 0 deg angle)

# 2. Test radiometric chain and FPA projection across a generation
python angle_sweep.py
# (Select Gen 1 to test block reduction, then Gen 2 to test interpolation)

# 3. Verify detector modeling
python fpa_diagnostic.py <any_psf_file.txt>

# 4. Check complete system
python full_fpa_demo.py <psf_file.txt> --magnitude 3.0 --trials 5
```

**Success Criteria**: 
- >95% detection success rates for magnitude 3.0 stars
- Centroiding accuracy <0.3 pixels (detector scale)
- Bearing vector accuracy <10 arcseconds
- Intensity conservation >99.8% during all projections

---

## Core Architecture Deep Dive

### 1. Radiometric Modeling (`starcamera_model.py`)

**Purpose**: Convert stellar magnitudes to photon counts accounting for all system losses.

**Key Function**: `calculate_optical_signal(star_obj, camera, scene)`

```python
# Calculation chain:
stellar_flux = star.photon_flux(passband)  # photons/s/m²
aperture_photons = stellar_flux * aperture_area  # photons/s
transmitted_photons = aperture_photons * transmission  # photons/s  
detected_photons = transmitted_photons * quantum_efficiency * integration_time  # photons
```

**Critical Implementation Details**:
- Temperature-dependent dark current: `dark_current = ref_current * 2^((T-T_ref)/coefficient)`
- Read noise is Gaussian, dark current and photon noise are Poisson
- Full well saturation is modeled but rarely reached in typical star tracker scenarios

**Maintenance Notes**:
- Camera parameters are set in `_get_default_camera()` 
- CMV4000 parameters are validated against datasheet specifications
- Scene parameters (temperature, integration time) significantly affect noise

### 2. PSF Processing and Projection

**File Parsing** (`psf_plot.py`):
- `discover_psf_generations_and_angles()`: **NEW** entry point. Scans the `PSF_sims/` directory to automatically find `Gen_X` subfolders and parse `Y_deg.txt` filenames. This populates the interactive user menus.
- `parse_psf_file()`: Robustly parses individual PSF files, handling variations in metadata formatting.

**Detector Projection** (`project_psf_to_fpa_grid()`):
This is the most critical update. The pipeline no longer assumes a single PSF format and instead uses a dynamic projection method.

```python
# 1. Calculate ideal scale factor
scale_factor = target_pixel_pitch_um / psf_pixel_spacing_um # e.g., 5.5 / 0.232 = 23.7 for Gen 2

# 2. Determine if PSF is too small for block reduction
min_required_pixels = int(np.round(scale_factor)) * 3
use_interpolation = original_psf_height < min_required_pixels

# 3. Choose projection method
if use_interpolation:
    # METHOD A: CUBIC INTERPOLATION (for small, high-res PSFs like Gen 2)
    # Upsamples the fine PSF grid to a physically-correct, small FPA grid (e.g., 3x3)
    # Preserves sub-pixel features when PSF is smaller than FPA pixels.
    fpa_intensity = scipy.ndimage.zoom(...) 
else:
    # METHOD B: BLOCK INTEGRATION (for large, low-res PSFs like Gen 1)
    # Sums NxN blocks of the original PSF into a single FPA pixel.
    # Perfectly models detector photon integration.
    fpa_intensity = skimage.measure.block_reduce(...)
```

**Why the Dual Method?**
- **Gen 1 PSFs** are large enough that their physical area covers many FPA pixels. **Block Integration** is the physically correct way to simulate how a real detector would sum the light falling on its pixels.
- **Gen 2 PSFs** are so physically small that their entire area fits within a couple of FPA pixels. **Block Integration** would crush the signal into a single, useless pixel. **Interpolation** correctly resamples the high-resolution PSF data onto a small grid that represents its true scale on the FPA.

### 3. Detection and Centroiding (`identify.py`, `star_tracker_pipeline.py`)

**Adaptive Thresholding for Diffuse PSFs**:
- **Problem**: PSFs at high field angles are diffuse. When projected, their signal is weak and spread out, causing them to be missed by a fixed detection threshold.
- **Solution**: The pipeline now analyzes the FPA-projected PSF's `spread_ratio`, `peak_to_mean_ratio`, and `concentration_ratio`. If these metrics indicate a diffuse PSF, the detection `k_sigma` is automatically lowered from `5.0` to a more sensitive `2.0-3.0`, ensuring robust detection.

**Adaptive Thresholding**:
```python
# Local statistics calculated over blocks
for i in range(num_blocks_y):
    for j in range(num_blocks_x):
        block = image[y_start:y_end, x_start:x_end]
        local_mean[i, j] = np.mean(block)
        local_std[i, j] = np.std(block)

# Threshold map upsampled to full image
threshold_map = mean_upsampled + k_sigma * std_upsampled
binary_image = (image > threshold_map).astype(np.uint8)
```

**Connected Components**: Uses OpenCV `connectedComponentsWithStats()` for speed and robustness.

**Region Selection Evolution**:
```python
# Original approach (unreliable):
closest_region = min(regions, key=lambda r: distance_to_true_center(r))

# Current approach (robust):
brightest_region = max(regions, key=lambda r: total_intensity(r))
```

**Why Brightness-Based Selection Works**: Physics dictates that the star is the brightest object in the field. Much more reliable than trying to predict spatial location.

**Adaptive Parameter Scaling**:
```python
# Automatically adjust for different grid sizes
fpa_grid_area = height * width

if fpa_grid_area < 200:  # Small grids (e.g., 11×11 = 121 pixels)
    min_pixels = 1
    max_pixels = max(20, fpa_grid_area // 5)
    block_size = max(4, min(adaptive_block_size, min(height, width) // 2))
else:  # Large grids (e.g., 128×128)
    min_pixels = 3
    max_pixels = 50
    block_size = adaptive_block_size
```

### 4. Bearing Vector Calculation

**Coordinate System Convention**:
- Image coordinates: (0,0) at top-left, x increases right, y increases down
- Physical coordinates: Origin at detector center
- Bearing vectors: Camera frame (Z optical axis, X/Y in focal plane)

**Implementation**:
```python
# Convert pixel coordinates to focal plane coordinates
center_x, center_y = width / 2.0, height / 2.0
focal_length_px = focal_length_mm * 1000 / pixel_pitch_um

X_focal_plane = (x - center_x) / focal_length_px
Y_focal_plane = (y - center_y) / focal_length_px

# Create unit vector [X, Y, Z] where Z is optical axis
vector = np.array([X_focal_plane, Y_focal_plane, 1.0])
unit_vector = vector / np.linalg.norm(vector)
```

**Critical Note**: This coordinate system is embedded throughout. Change at your peril.

---

## Implementation Gotchas and Lessons Learned

### 1. PSF Metadata Parsing

**The Problem**: Inconsistent PSF file formats and metadata.

**Robust Solution**:
```python
def _get_psf_data_spacing_microns(self, psf_metadata):
    default_spacing_um = 0.500  # Known good for our files
    
    if psf_metadata and 'data_spacing' in psf_metadata:
        try:
            # Handle multiple format possibilities
            data_spacing_value = psf_metadata['data_spacing']
            if isinstance(data_spacing_value, (float, int)):
                return float(data_spacing_value)
            elif isinstance(data_spacing_value, str):
                # Parse "0.5000 um" format
                spacing_str = data_spacing_value.split()[0]
                return float(spacing_str)
        except (ValueError, IndexError, AttributeError):
            logger.warning(f"Could not parse data spacing, using default {default_spacing_um} µm")
    
    return default_spacing_um
```

**Maintenance Tip**: When adding new PSF file formats, enhance this method first. All scaling calculations depend on correct pixel spacing.

### 2. Intensity Conservation Validation

**The Critical Invariant**: `np.sum(original_psf) ≈ np.sum(projected_psf)` within 0.2%

```python
# Always validate in projection functions
original_total = np.sum(psf_intensity)
projected_total = np.sum(projected_intensity)
conservation_ratio = projected_total / original_total if original_total > 0 else 0

if conservation_ratio < 0.998:
    logger.warning(f"Poor intensity conservation: {conservation_ratio:.4f}")
    # This usually indicates PSF data spacing parsing failure
```

**Debug Strategy**: Poor conservation almost always traces to incorrect PSF pixel spacing.

### 3. Memory Management for Large Arrays

**Full Detector Arrays**: 2048×2048 float64 arrays consume ~33MB each.

**Best Practices**:
```python
# For batch processing
for psf_file in large_psf_list:
    results = process_psf(psf_file)
    save_results(results)
    
    # Explicitly clean up
    if 'full_fpa_intensity' in results:
        del results['full_fpa_intensity']
    gc.collect()

# Lazy creation pattern
def get_full_detector_array(self):
    if not hasattr(self, '_full_detector_cache'):
        self._full_detector_cache = create_full_detector()
    return self._full_detector_cache
```

### 4. Random Placement and Reproducibility

**Design Decision**: Random PSF placement on full detector (no fixed seed).

**Rationale**: Real stars appear at arbitrary positions. Random placement tests algorithm robustness and provides realistic statistics.

**When You Need Reproducibility**:
```python
# Add at beginning of script
np.random.seed(42)
```

**Note**: Randomness is typically desired for Monte Carlo analysis. Only set seed for debugging.

---

## Testing Strategy and Validation

### Validation Hierarchy

**Level 1: Physics Validation**
- **Intensity Conservation**: Must be >99.8% for both block reduction and interpolation methods.
- **Projection Method Choice**: Manually verify that `debug_centroiding.py` selects 'interpolation' for Gen 2 and 'block_reduction' for Gen 1.

**Level 2: Algorithm Validation**
- **Diffuse PSF Detection**: Use `debug_fpa_issue.py` on 12° and 14° Gen 1 PSFs to confirm the adaptive threshold is triggered and lowers the sigma value.
- **Detection Success Rates**: Should be >95% for all tested angles and generations with a bright star.

**Level 3: System Integration**
```bash
# End-to-end pipeline test
python angle_sweep.py --magnitude 3.0 --trials 50
# Expected: Consistent performance across field angles
```

### Debugging Workflow

**Problem: "Poor performance"**

1. **Check fundamental physics**: Intensity conservation, photon counts
2. **Validate PSF parsing**: Correct pixel spacing and total intensity
3. **Examine detection parameters**: Thresholds appropriate for signal level
4. **Compare analysis modes**: High-resolution vs detector-projected
5. **Visualize intermediate steps**: Use diagnostic tools

**Problem: "No detections"**

1. **Check photon count**: Try brighter star (magnitude 1.0-2.0)
2. **Lower detection threshold**: `threshold_sigma=3.0` instead of 5.0
3. **Verify PSF quality**: Not corrupted, reasonable intensity distribution
4. **Check grid size**: Very small PSFs may not project well

**Problem: "Inconsistent results"**

1. **Random seeds**: Set for reproducibility if needed
2. **Parameter consistency**: Verify same settings between runs
3. **File parsing**: Ensure PSF metadata parsed identically
4. **Temperature/noise**: Check if environmental parameters vary

---

## Architecture Extension Points

### Adding New Detectors

**Required Changes**:
1. **Pixel pitch**: Modify `target_pixel_pitch_um` parameter
2. **Detector size**: Modify `fpa_size` parameter
3. **Scale factor validation**: Ensure close to integer

**Implementation Pattern**:
```python
DETECTOR_SPECS = {
    'CMV4000': {'pixel_pitch': 5.5, 'size': (2048, 2048)},
    'IMX421': {'pixel_pitch': 4.5, 'size': (5536, 3692)},
    'CMV12000': {'pixel_pitch': 5.5, 'size': (4096, 3072)}
}

def analyze_for_detector(psf_data, detector_type='CMV4000'):
    specs = DETECTOR_SPECS[detector_type]
    return run_monte_carlo_simulation_fpa_projected(
        psf_data,
        target_pixel_pitch_um=specs['pixel_pitch'],
        fpa_size=specs['size']
    )
```

### Adding New Analysis Modes

**Follow the Dual Analysis Pattern**:
```python
def run_new_analysis(self, psf_data, **kwargs):
    """Template for new analysis modes"""
    
    # 1. Run both analysis modes
    high_res_results = self.run_monte_carlo_simulation(psf_data, **kwargs)
    detector_results = self.run_monte_carlo_simulation_fpa_projected(psf_data, **kwargs)
    
    # 2. Extract metrics of interest
    high_res_metric = extract_metric(high_res_results)
    detector_metric = extract_metric(detector_results)
    
    # 3. Generate comparison and validation
    comparison = {
        'high_resolution': high_res_metric,
        'detector_projected': detector_metric,
        'physics_check': validate_physics_consistency(high_res_results, detector_results)
    }
    
    # 4. Visualize and return
    if kwargs.get('create_plots', True):
        create_comparison_plots(comparison)
    
    return comparison
```

### Adding New PSF File Formats

**Required Changes**:
1. **Update `parse_psf_file()`** in `psf_plot.py`
2. **Enhance metadata parsing** in `_get_psf_data_spacing_microns()`
3. **Add format detection** if file extensions differ

**Template**:
```python
def parse_new_psf_format(file_path):
    """Parse new PSF file format"""
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract metadata
    metadata = extract_metadata(content)
    
    # Parse intensity data
    intensity_data = extract_intensity_array(content)
    
    # Validate
    assert intensity_data.ndim == 2, "PSF must be 2D array"
    assert np.sum(intensity_data) > 0, "PSF must have positive intensity"
    assert 'data_spacing' in metadata, "Must include pixel spacing"
    
    return metadata, intensity_data
```

---

## Performance Optimization and Scalability

### Current Performance Characteristics

**Timing** (on modern laptop):
- Single PSF analysis: 2-5 seconds
- Field angle sweep (11 angles): 2-3 minutes
- Monte Carlo overhead: ~0.1 seconds per trial

**Memory Usage**:
- Base pipeline: ~50MB
- High-resolution PSF arrays: ~130KB each (128×128 float64)
- Detector arrays: ~1KB each (11×11 float64)  
- Full detector arrays: ~33MB each (2048×2048 float64)

### Optimization Opportunities

**1. Parallel Monte Carlo Trials**
```python
# Current: Serial processing
results = []
for trial in range(num_trials):
    result = simulate_trial(trial_params)
    results.append(result)

# Future: Parallel processing
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(simulate_trial, trial_param_list)
```

**2. PSF File Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=50)
def load_psf_cached(file_path):
    return parse_psf_file(file_path)
```

**3. Vectorized Operations**
```python
# Current: Loop over centroids
for x, y in centroids:
    vector = calculate_bearing_vector(x, y, camera_params)

# Future: Vectorized
centroids_array = np.array(centroids)
vectors = calculate_bearing_vectors_vectorized(centroids_array, camera_params)
```

### Scalability Considerations

**Large Batch Processing**:
- Process PSF files individually to avoid memory accumulation
- Use generators for large file lists
- Implement progress reporting for long-running analyses

**Memory-Constrained Environments**:
- Disable full detector visualization by default
- Use float32 instead of float64 where precision allows
- Implement lazy loading of large arrays

---

## Technical Debt and Future Priorities

### High Priority (Should Address Soon)

**1. Unit Test Suite**
- **Priority**: High - enables confident refactoring
- **Effort**: 2-3 weeks full-time
- **Coverage**: Core physics calculations, PSF parsing, centroiding algorithms

**2. Configuration System**
- **Current**: Hard-coded parameters scattered throughout code
- **Future**: YAML/JSON config files for camera specs, algorithm parameters
- **Value**: Easier parameter tuning, better reproducibility

**3. Error Handling Improvements**
- **Current**: Graceful degradation with warnings
- **Future**: More specific error messages and recovery strategies
- **Value**: Easier debugging for new users

### Medium Priority (Nice to Have)

**1. Performance Optimization**
- Parallel Monte Carlo processing
- PSF file caching
- Vectorized operations where beneficial

**2. Interactive Visualization**
- **Current**: Static matplotlib plots
- **Future**: Interactive Jupyter widgets or web interface
- **Value**: Better parameter space exploration

**3. Advanced Detector Effects**
- Hot pixels, dead pixels, non-linearity
- More sophisticated noise models
- Detector aging effects

### Low Priority (Future Enhancement)

**1. Alternative Centroiding Algorithms**
- Gaussian fitting
- Iterative approaches
- Machine learning methods

**2. Multi-Star Scenarios**
- Multiple stars in field
- Crowded field analysis
- Star catalog integration

### Technical Debt to Avoid

**1. Don't Optimize Prematurely**
- Current performance adequate for most use cases
- Profile before optimizing

**2. Don't Break Backwards Compatibility**
- Existing users depend on current API
- Deprecate gracefully if changes needed

**3. Don't Remove Dual Analysis**
- Comparison capability is core value proposition
- Validation requires both modes

---

## Integration and Dependency Management

### Core Dependencies

**NumPy/SciPy**: 
- Used throughout for array operations and scientific computing
- **Version Sensitivity**: Low (any recent version works)
- **Critical Functions**: Array operations, statistical functions, FFT (unused currently)

**Matplotlib**:
- All visualization and plotting
- **Version Sensitivity**: Medium (newer versions have better features)
- **Critical Functions**: `imshow()`, `plot()`, `subplots()`, colormaps

**OpenCV**:
- Connected components analysis only
- **Version Sensitivity**: Low (any cv2 version works)
- **Critical Functions**: `connectedComponentsWithStats()`
- **Alternative**: Could use `scipy.ndimage.label` but would lose performance

**Pandas**:
- CSV output and data organization
- **Version Sensitivity**: Low
- **Critical Functions**: DataFrame creation, CSV export

### Integration Points

**Input Interfaces**:
- PSF file parsing: Extensible to new formats
- Camera parameter configuration: Programmatic setup
- Scene parameter configuration: Runtime adjustment

**Output Interfaces**:
- Consistent result dictionary structure across all analysis modes
- CSV export for external analysis tools
- PNG visualization for reports

**Extension Points**:
- New detector types: Parameter-driven configuration
- New analysis modes: Template-based implementation
- New file formats: Parser extension

---

## Development Best Practices

Please follow standard SOE best-practices for coding (use standard pep-8). Consider installing a Linter extension to VSCode to make your formatting life easier.

### Performance Considerations

**Memory Management**:
```python
# Explicit cleanup for large arrays
del large_array
gc.collect()

# Use appropriate data types
array_float32 = np.array(data, dtype=np.float32)  # For visualization
array_float64 = np.array(data, dtype=np.float64)  # For calculations

# Avoid unnecessary copies
view = array[start:end]  # View, not copy
copy = array[start:end].copy()  # Explicit copy when needed
```

**Algorithmic Efficiency**:
```python
# Vectorized operations
result = np.sum(array1 * array2)  # Not: sum([a*b for a,b in zip(array1, array2)])

# Pre-allocation
results = np.zeros((num_trials, 3))  # Not: results = []

# Appropriate algorithms
# Use scipy.ndimage for image processing when OpenCV not required
```

### Debugging Strategies

**Enable Comprehensive Logging**:
```python
# Enable debug mode for detailed output
pipeline = StarTrackerPipeline(debug=True)
logging.getLogger().setLevel(logging.DEBUG)
```

**Validate Invariants Early**:
```python
# Check critical assumptions
assert np.sum(psf_intensity) > 0, "PSF must have positive intensity"
assert conservation_ratio > 0.998, f"Poor conservation: {conservation_ratio:.4f}"
assert psf_shape[0] == psf_shape[1], "PSF must be square"
```

**Visual Debugging**:
```python
# Plot intermediate results for debugging
def debug_plot(data, title="Debug"):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='hot', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Use in development
debug_plot(psf_intensity, "Original PSF")
debug_plot(projected_intensity, "Projected PSF")
```

## How to Add a New PSF Generation

This system is designed for extension. To add a new generation of PSF data:

**1. Add Files to `PSF_sims/` Directory**
- Create a new folder: `PSF_sims/Gen_3/`
- Place PSF files inside, following the `[angle]_deg.txt` naming convention (e.g., `0_deg.txt`, `7.5_deg.txt`).

**2. Run Analysis Scripts**
- The interactive scripts (`debug_centroiding.py`, `angle_sweep.py`) will automatically discover "Generation 3" and its corresponding angles.
- Select the new generation from the menu to run a full analysis.

**3. The Pipeline Handles the Rest**
- The `project_psf_to_fpa_grid` function will automatically assess the new PSF's grid size and data spacing.
- It will choose the appropriate projection method (Block Reduction or Interpolation) to ensure a physically accurate simulation on the CMV4000 FPA model. No code changes are needed for new generations that follow the file structure convention.
