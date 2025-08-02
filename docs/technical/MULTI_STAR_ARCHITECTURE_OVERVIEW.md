# Multi-Star Pipeline Architecture Overview

## System Flow Diagram

```
Synthetic Catalog Generation
         ↓
    Scene Generation (RA/Dec → FPA coordinates)
         ↓
    Multi-Star Radiometry (PSF rendering + Poisson noise)
         ↓
    Star Detection (Peak detection method)
         ↓
    Bearing Vector Calculation
         ↓
    BAST Triangle Matching
         ↓
    Validation & Debug Analysis
```

## Data Flow Architecture

### 1. Synthetic Catalog (`synthetic_catalog.py`)
**Input**: Configuration parameters (separation, magnitude)
**Output**: BAST-compatible catalog with triplet data
**Key Functions**: `create_triangle_catalog()`, `create_pyramid_catalog()`

### 2. Scene Generator (`scene_generator.py`)
**Input**: Synthetic catalog
**Output**: Scene data with FPA coordinates
**Key Functions**: `generate_scene()`, `_sky_to_detector()`
**Critical**: Converts RA/Dec to pixel coordinates

### 3. Multi-Star Radiometry (`multi_star_radiometry.py`)
**Input**: Scene data + PSF data
**Output**: Combined detector image
**Key Functions**: `render_scene()`, `_calculate_detector_canvas_size()`
**Critical**: Dynamic sizing for multi-star scenes

### 4. Star Detection (`peak_detection.py`)
**Input**: Detector image
**Output**: Star centroids
**Key Functions**: `detect_stars_peak_method()`
**Critical**: Superior to adaptive thresholding for multi-star scenes

### 5. Pipeline Integration (`multi_star_pipeline.py`)
**Input**: Catalog + PSF data
**Output**: Complete analysis results
**Key Functions**: `run_multi_star_analysis()`
**Critical**: Orchestrates entire multi-star workflow

## Key Design Decisions

### 1. Non-Destructive Extension
- Multi-star functionality extends existing pipeline without modification
- All analysis still flows through `StarTrackerPipeline` class
- Maintains backward compatibility with single-star analysis

### 2. Dual Analysis Paths
- **Original PSF**: High-resolution (128×128, 0.5μm/pixel) for optical design
- **FPA Projected**: Realistic detector simulation for hardware validation
- Both paths use same underlying algorithms

### 3. Validation-First Architecture
- Comprehensive validation at every stage
- Debug mode provides detailed analysis
- Coordinate transformations validated to preserve angular relationships

### 4. Modular Component Design
- Each component has single responsibility
- Clear interfaces between components
- Easy to extend or modify individual components

## Critical Integration Points

### 1. BAST Interface
- **File**: `BAST/match.py`
- **Critical Fix**: Angle conversion to degrees (lines 123-125)
- **Input**: Bearing vectors + catalog
- **Output**: Star matches with confidence

### 2. StarTrackerPipeline Integration
- **File**: `star_tracker_pipeline.py`
- **Role**: Central orchestrator for all analysis
- **Critical**: All analysis must flow through this class
- **Methods**: `detect_stars_and_calculate_centroids()`, `calculate_bearing_vectors()`

### 3. Coordinate System Consistency
- **RA/Dec**: Input celestial coordinates
- **FPA**: Focal plane array pixel coordinates
- **Bearing Vectors**: Unit vectors in camera frame
- **Critical**: Transformations preserve angular relationships

## Performance Architecture

### 1. Detection Performance
- **Method**: Peak detection with maximum filtering
- **Threshold**: Dynamic based on image statistics
- **Accuracy**: 0.15-0.25 pixels (0.8-1.4μm)
- **Success Rate**: 100% for 3-star scenarios

### 2. Coordinate Accuracy
- **Transformation Error**: <0.001° throughout pipeline
- **Angular Preservation**: Validated at each stage
- **Bearing Vector Precision**: 6 decimal places

### 3. BAST Matching Performance
- **Confidence**: 1.000 for validated scenarios
- **Tolerance**: 5° angle tolerance (configurable)
- **Speed**: <1 second for 3-star matching

## Extensibility Architecture

### 1. New Star Configurations
- **Current**: 3-star triangle (fully validated)
- **Next**: 4-star pyramid (framework exists)
- **Future**: N-star arbitrary configurations

### 2. New PSF Generations
- **Auto-Detection**: System detects new PSF generations
- **Dynamic Loading**: Adapts to different PSF grid sizes
- **Extensible**: Easy to add new PSF formats

### 3. New Validation Methods
- **Framework**: Modular validation system
- **Current**: Coordinate validation, match validation
- **Future**: Performance validation, stress testing

## Debug Architecture

### 1. Debug Mode Infrastructure
- **Activation**: `--debug` command line flag
- **Scope**: Comprehensive analysis of entire pipeline
- **Output**: Detailed logging of all critical parameters

### 2. Debug Information Categories
- **Coordinates**: Pixel and physical coordinates
- **Angles**: Calculated vs catalog angle comparison
- **Matching**: Detailed match analysis
- **Validation**: Ground truth comparison

### 3. Debug Utilities
- **Files**: `coordinate_validation.py`, `debug_star_detection.py`
- **Functions**: `log_debug_matching_details()`, `log_coordinate_pipeline_details()`
- **Integration**: Seamlessly integrated into main pipeline

## Quality Assurance Architecture

### 1. Validation Framework
- **Coordinate Validation**: Angular relationship preservation
- **End-to-End Testing**: Complete pipeline validation
- **Performance Metrics**: Quantitative assessment
- **Debug Analysis**: Comprehensive process analysis

### 2. Error Detection
- **Unit Consistency**: Automatic unit validation
- **Coordinate Integrity**: Transformation accuracy checks
- **Algorithm Correctness**: Match confidence analysis
- **Performance Monitoring**: Success rate tracking

### 3. Success Criteria
- **Star Detection**: 100% success rate
- **Coordinate Accuracy**: <0.001° error
- **BAST Matching**: >0.9 confidence
- **Pipeline Integrity**: All validations pass

This architecture enables reliable, extensible, and thoroughly validated multi-star analysis.