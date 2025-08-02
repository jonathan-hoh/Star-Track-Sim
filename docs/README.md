# Star Tracker Radiometry Pipeline

A comprehensive simulation and analysis pipeline for star tracker performance evaluation, featuring complete radiometric modeling from optical Point Spread Functions (PSFs) through detector response, centroiding algorithms, and bearing vector calculation.

## Overview

This pipeline simulates the complete star tracker signal chain: optical PSFs → photon detection → detector response → star identification → centroiding → bearing vector calculation. It provides tools for analyzing how various parameters affect star tracker performance and enables optimization of both optical and algorithmic components.

## Key Features

- **Complete Radiometric Modeling**: From PSF files through photon statistics to detector output
- **Realistic Detector Simulation**: Models actual focal plane array (FPA) behavior including the CMV4000 sensor
- **Advanced Centroiding**: Sub-pixel accuracy with adaptive thresholding and brightness-based region selection
- **Monte Carlo Analysis**: Statistical evaluation across parameter spaces with confidence intervals
- **Bearing Vector Calculation**: 3D unit vectors with angular accuracy metrics
- **Comprehensive Analysis Tools**: Field angle sweeps, magnitude limits, temperature effects, optical parameter optimization
- **Multiple Analysis Modes**: Both high-resolution PSF analysis and realistic detector projection
- **Flexible Output**: Results in both pixel units and physical units (µm, arcsec) with CSV and visualization

## System Capabilities

### Radiometric Chain Simulation
```
Stellar Flux → Optical System → PSF → Detector → Digital Counts → Centroiding → Bearing Vectors
```

The pipeline models each step with realistic physics:
- **Stellar photometry**: Magnitude-based photon flux calculation
- **Optical modeling**: Transmission, f-stop, focal length effects  
- **PSF processing**: From Zemax simulations to detector-scale projections
- **Detector physics**: Quantum efficiency, noise, saturation, temperature effects
- **Image processing**: Adaptive thresholding, connected components, moment-based centroiding

### Analysis Modes

**High-Resolution PSF Analysis**: Direct analysis of optical PSF simulations (128×128, 0.5µm/pixel)
- Ideal for optical design optimization
- High precision for algorithm development
- Academic research and validation

**Detector-Projected Analysis**: Realistic detector simulation (11×11 CMV4000 pixels, 5.5µm/pixel)  
- Hardware-accurate performance predictions
- Accounts for pixel integration effects
- Production system validation

**Full Detector Context**: Complete focal plane visualization (2048×2048 pixels)
- True hardware scale understanding
- Random star placement simulation
- System-level performance assessment

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Clone and Install

```bash
git clone https://github.com/yourusername/radiometry-pipeline.git
cd radiometry-pipeline
pip install -r config/requirements.txt
```

Required packages:
```bash
pip install numpy scipy matplotlib pandas opencv-python
```

## Quick Start

### Interactive Analysis

```bash
# Run interactive PSF debugging
PYTHONPATH=. python tools/interactive/debug_centroiding.py

# Analyze performance across field angles  
PYTHONPATH=. python tools/interactive/angle_sweep.py

# Diagnostic analysis of specific PSF
PYTHONPATH=. python tools/interactive/fpa_diagnostic.py data/PSF_sims/Gen_1/0_deg.txt
```

### Specialized Analysis

```bash
# Full system demonstration
PYTHONPATH=. python tools/specialized/full_fpa_demo.py data/PSF_sims/Gen_1/0_deg.txt --magnitude 3.0 --trials 5

# Analyze PSF quality across field angles
PYTHONPATH=. python tools/specialized/analyze_psf_quality.py --psf-dir data/PSF_sims/Gen_1/
```

### Directory Structure

```
radiometry-pipeline/
├── src/                     # Core source code
│   ├── core/                # Main pipeline components
│   ├── multi_star/          # Multi-star functionality  
│   └── BAST/                # Star identification algorithms
├── tools/                   # Analysis scripts
│   ├── interactive/         # Primary user tools
│   ├── specialized/         # Advanced analysis tools
│   └── debug/               # Diagnostic tools
├── data/                    # Input data
│   ├── PSF_sims/           # PSF simulation files
│   ├── examples/           # Example datasets
│   └── tests/              # Test data
├── docs/                    # Documentation
├── outputs/                 # Generated results
└── config/                  # Configuration files
```

## Quick Start

### Basic Analysis

The pipeline is now managed through interactive scripts that guide the user through selecting a PSF generation and field angle.

```bash
# Interactive analysis of a single PSF with performance comparison
python debug_centroiding.py

# Complete field angle sweep across a chosen PSF generation
python angle_sweep.py

# Comprehensive demonstration of the full pipeline
# (Note: This script may need updates for the new PSF structure)
# python full_fpa_demo.py
```

## Core Scripts and Usage

### Analysis Scripts

#### `star_tracker_pipeline.py` - Main Analysis Pipeline
Complete radiometric simulation with multiple analysis modes.

```bash
python star_tracker_pipeline.py --psf PSF_FILE [options]

# Key options:
--magnitude FLOAT         # Star magnitude (default: 3.0)
--photon-count FLOAT      # Direct photon specification  
--simulations INT         # Monte Carlo trials (default: 50)
--f-stop FLOAT           # Optical f-stop (default: 1.7)
--focal-length FLOAT     # Focal length in mm (default: 32.0)
--temperature FLOAT      # Detector temperature (default: 20.0)
--debug                  # Enable detailed logging
```

#### `angle_sweep.py` - Field Angle Analysis
Performs a comprehensive analysis across all available field angles for a user-selected PSF generation.

```bash
# Launch the interactive angle sweep
python angle_sweep.py

# Options:
--output DIR            # Output directory (default: angle_sweep_outputs/Gen_X_Results)
--magnitude FLOAT       # Star magnitude
--trials INT           # Monte Carlo trials per angle
```

#### `debug_centroiding.py` - Interactive Analysis and Debugging
Provides a detailed comparison between the original high-resolution PSF analysis and the realistic FPA-projected analysis. It now interactively prompts for PSF generation and field angle.

```bash
# Launch the interactive debugger
python debug_centroiding.py

# Options still available:
--magnitude FLOAT       # Star magnitude (default: 3.0)
--trials INT           # Number of trials (default: 10)
--full-fpa             # Show complete 2048×2048 detector visualization
```

#### `full_fpa_demo.py` - Comprehensive System Demonstration
Complete visualization of the radiometry pipeline from PSF to bearing vectors.

```bash
python full_fpa_demo.py PSF_FILE [options]

python full_fpa_demo.py 8_49deg_PSF_32mmEFL_F1_7_FieldFlattener_FPA1_Wideband_Option2_DFM.txt --magnitude 3.0 --trials 5

# Options:
--magnitude FLOAT       # Star magnitude (default: 3.0)  
--trials INT           # Monte Carlo trials (default: 5)
--output DIR           # Output directory (default: full_fpa_demo)
```

#### `fpa_diagnostic.py` - Detailed Technical Analysis
Step-by-step analysis of the detector projection and signal processing chain.

```bash
python fpa_diagnostic.py PSF_FILE

# Provides detailed diagnostics of:
# - PSF characteristics and quality
# - Detector projection process
# - Signal detection parameters
# - Processing step validation
```

### Specialized Analysis Scripts

#### `focal_length_analysis.py` - Optical Design Optimization
```bash
python focal_length_analysis.py --focal-lengths 20 25 30 35 40 45 50
```

#### `analyze_psf_quality.py` - PSF Characterization
```bash
python analyze_psf_quality.py --psf-dir PSF_DIRECTORY
```

## Understanding the Output

### Performance Metrics

#### Centroiding Performance
- **Pixel accuracy**: Sub-pixel centroiding precision 
- **Physical accuracy**: Real-world measurement precision in micrometers
- **Success rate**: Percentage of successful star detections

#### Bearing Vector Performance
- **Angular accuracy**: Measured in arcseconds
- **3D precision**: Full unit vector accuracy
- **Field angle dependence**: Performance variation across sensor

#### System Performance
- **Detection reliability**: Success rates across conditions
- **Noise tolerance**: Performance vs background noise and detector temperature
- **Magnitude limits**: Faintest detectable stars

### Typical Performance Expectations

**Magnitude 3.0 star, standard conditions**:
- Centroiding accuracy: 0.15-0.25 detector pixels (0.8-1.4 µm)
- Bearing vector accuracy: 4-8 arcseconds
- Detection success rate: >95% (field angles 0-14°)
- Processing time: 2-5 seconds per analysis

### Output Files

```
analysis_results/
├── comparison_plots.png              # Performance visualizations
├── field_angle_sweep.csv            # Numerical results across angles
├── diagnostic_plots.png             # Technical analysis plots
└── performance_summary.txt          # Key metrics and statistics
```

## CMV4000 Sensor Specifications

The pipeline is optimized for the CMV4000 global shutter CMOS sensor:

- **Resolution**: 2048 × 2048 pixels
- **Pixel Pitch**: 5.5 µm × 5.5 µm  
- **Optical Area**: 11.3 mm × 11.3 mm
- **Quantum Efficiency**: ~60% (configurable)
- **Full Well Capacity**: 13,500 e⁻
- **Read Noise**: 13.0 e⁻ RMS

## Advanced Configuration

### Camera Model Parameters
```python
# Optical system configuration
optic = star_camera.optic_stack(
    f_stop=1.7,              # f-stop
    ffov=16.0,               # Full field of view (degrees)
    transmission=0.8         # Optical transmission
)

# Detector configuration  
fpa = star_camera.fpa(
    x_pixels=2048,           # Detector width
    y_pixels=2048,           # Detector height
    pitch=5.5,               # Pixel pitch (µm)
    qe=0.6,                  # Quantum efficiency
    dark_current_ref=125,    # Dark current (e⁻/s)
    full_well=13500,         # Full well capacity (e⁻)
    read_noise=13.0          # Read noise (e⁻ RMS)
)
```

### Scene Parameters
```python
scene_params = scene(
    int_time=17.0,           # Integration time (ms)
    temp=20.0,               # Temperature (°C)
    slew_rate=0.1,           # Slew rate (°/s)
    fwhm=2.0                 # Expected PSF FWHM (pixels)
)
```

### Detection Algorithm Parameters
```python
# Centroiding parameters
threshold_sigma = 5.0        # Detection threshold (σ above background)
adaptive_block_size = 16     # Local thresholding block size
min_pixels = 3               # Minimum star region size
max_pixels = 50              # Maximum star region size
```

## Troubleshooting

### Common Issues

**No detections found**:
- Try brighter stars (lower magnitude numbers): `--magnitude 2.0`
- Reduce detection threshold: `threshold_sigma=4.0`
- Check PSF file format and data spacing

**Poor centroiding accuracy**:
- Verify sufficient photon count for star magnitude
- Check detector temperature settings
- Ensure PSF files have correct metadata

**Inconsistent results**:
- Use fixed random seed for reproducibility: `np.random.seed(42)`
- Verify input parameters are consistent between runs
- Check for proper PSF file parsing
- Increasing the number of `simulations` in the Monte Carlo function should bring you closer to large-number approximations 

### Debug Tools

```bash
# Enable detailed logging
python debug_centroiding.py psf_file.txt --magnitude 3.0 --debug

# Step-by-step analysis
python fpa_diagnostic.py psf_file.txt

# Performance validation
python angle_sweep.py --magnitude 3.0 --trials 100
```

### Expected Warning Messages

**Linter warnings about OpenCV**: `cv2` member access warnings are cosmetic - functionality works correctly.

**PSF data spacing warnings**: Default values used when metadata parsing fails - usually acceptable.

**Memory usage warnings**: Large detector arrays (2048×2048) use ~33MB each - normal for full detector visualization.


## File Structure

```
radiometry-pipeline/
├── PSF_sims/                        # Root directory for all PSF data
│   ├── Gen_1/                       # PSF files for Generation 1
│   │   ├── 0_deg.txt
│   │   └── ...
│   └── Gen_2/                       # PSF files for Generation 2
│       └── ...
├── star_tracker_pipeline.py         # Main analysis pipeline
├── angle_sweep.py                   # Field angle analysis
├── debug_centroiding.py             # Interactive debugging
├── psf_plot.py                      # PSF file parsing and visualization
├── starcamera_model.py              # Radiometric camera modeling
└── ...                              # Other scripts and documentation
```

## Performance and Scalability

### Processing Speed
- Single PSF analysis: 2-5 seconds
- Field angle sweep (11 angles): 2-3 minutes  
- Monte Carlo trials scale linearly with trial count

### Memory Usage
- Base analysis: <100MB RAM
- Full detector visualization: +33MB per detector array
- Batch processing: Memory usage scales with concurrent PSF files

### Optimization Recommendations
- Use detector-projected analysis for hardware predictions
- Run high-resolution PSF analysis for optical design
- Enable full detector visualization only when needed for scale context

## Contributing and Extension

### Adding New Detectors
The system is designed to support different detector types by modifying pixel pitch and array size parameters.

### Adding New Analysis Modes
Follow the dual-analysis pattern: implement both high-resolution PSF and detector-projected versions for consistency.

### PSF File Formats
The PSF parser supports standard Zemax PSF output formats. New formats can be added by extending the parsing functions.

## Support and Documentation

- **User Guide**: This README
- **System Architecture**: See ARCHITECTURE.md  
- **Developer Guide**: See HANDOFF_GUIDE.md
- **Function Documentation**: Inline docstrings in source code

For technical questions about star tracker physics, detector modeling, or optical design integration, refer to the architecture documentation and developer handoff guide. For specific questions, talk to Dr. J in person. 

## How to Add and Analyze a New PSF Generation

To extend the pipeline with a new generation of PSF data (e.g., from a new optical design), follow these steps:

**1. Directory Structure**

Create a new subfolder inside the `PSF_sims/` directory named `Gen_X`, where `X` is the new generation number.

```
radiometry-pipeline/
└── PSF_sims/
    ├── Gen_1/
    │   ├── 0_deg.txt
    │   └── ...
    ├── Gen_2/
    │   ├── 0_deg.txt
    │   └── ...
    └── Gen_3/   <-- ADD NEW FOLDER HERE
        ├── 0_deg.txt
        ├── 5_deg.txt
        └── 10_deg.txt
```

**2. PSF File Naming**

Place your new PSF simulation files inside the `Gen_X` folder. Ensure they follow the standardized naming convention: `[angle]_deg.txt` (e.g., `7.5_deg.txt`). The scripts will automatically discover and parse these files.

**3. Run Analysis**

The analysis scripts will automatically detect the new generation. Simply run them and select your new generation from the interactive menu.

```bash
# The new generation will appear in the selection menu
python debug_centroiding.py 
python angle_sweep.py
```

The system will handle different PSF grid sizes, pixel spacings, and physical areas, automatically choosing the correct projection method (block reduction or interpolation) for accurate FPA simulation. 