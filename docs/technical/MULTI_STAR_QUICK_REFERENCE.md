# Multi-Star Pipeline Quick Reference

## Essential Commands

### Validate Current System
```bash
# Verify everything is working
python3 multi_star/end_to_end_test.py --debug

# Quick validation without debug
python3 multi_star/end_to_end_test.py
```

### Key Development Files to Know
- **`multi_star/end_to_end_test.py`** - Main validation script
- **`multi_star/coordinate_validation.py`** - Debug and validation functions
- **`multi_star/multi_star_pipeline.py`** - Core pipeline integration
- **`BAST/match.py`** - Triangle matching (CRITICAL: uses degrees, not radians)

### Critical Bug Fixes Applied
1. **BAST Unit Conversion**: `BAST/match.py` lines 123-125 convert angles to degrees
2. **Dynamic Canvas Sizing**: `multi_star_radiometry.py` auto-sizes detector for multi-star scenes
3. **Peak Detection**: `peak_detection.py` handles closely-spaced stars better than adaptive thresholding

### System Status Check
```bash
# Expected output should show:
# ✅ Catalog stars: 3
# ✅ Detected stars: 3
# ✅ Coordinate validation: PASSED
# ✅ BAST matches: 1 (confidence 1.000)
# ✅ Overall test: PASSED
```

## Next Development Targets

### 1. Multi-Star Visualization (`multi_star_visualization.py`)
- Create visual debugging tools for multi-star scenes
- Show star positions, PSF overlays, detection results
- Validate detection performance visually

### 2. 4-Star Pyramid Validation
- Extend from 3-star triangle to 4-star pyramid
- Implement in `validation.py` (placeholder exists)
- Add to end-to-end test framework

### 3. Performance Analysis
- Monte Carlo analysis across different configurations
- Parameter optimization for BAST matching
- Field-dependent PSF integration

## Architecture Rules

### DO NOT MODIFY
- **StarTrackerPipeline class** - All analysis flows through this
- **BAST angle units** - Confirmed working with degrees
- **Coordinate transformation pipeline** - Validated to preserve angular relationships

### ALWAYS VERIFY
- Run `--debug` mode for comprehensive analysis
- Check coordinate transformation accuracy (<0.001° error)
- Ensure BAST matching confidence >0.9

### KNOWN WORKING STATE
- 3-star equilateral triangle, 0.5° separation
- Gen_1 PSF, 0-degree field angle
- Peak detection method for star identification
- CMV4000 sensor model (2048×2048, 5.5μm pixels)

## Emergency Debugging

### If Tests Fail
1. Check `BAST/match.py` angle conversion (lines 123-125)
2. Verify PSF file availability in `PSF_sims/Gen_1/`
3. Check coordinate transformation accuracy in debug output
4. Ensure peak detection finds all stars

### If No Stars Detected
1. Check PSF file loading and rendering
2. Verify detection thresholds in `peak_detection.py`
3. Check detector canvas sizing in `multi_star_radiometry.py`
4. Debug with `debug_star_detection.py`

### If BAST Matching Fails
1. Verify angle units are in degrees
2. Check coordinate transformation preserves angular relationships
3. Ensure bearing vectors are calculated correctly
4. Use debug mode to compare catalog vs observed angles

This quick reference provides the essential information needed to continue development efficiently.