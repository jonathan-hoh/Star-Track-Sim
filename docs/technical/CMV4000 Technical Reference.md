
# CMV4000 Technical Reference for Star Tracker Design

## Overview

The CMV4000 is a high-sensitivity, pipelined global shutter CMOS image sensor with 2048 x 2048 pixel resolution, designed for machine vision applications. This document summarizes key technical parameters relevant for star tracker design.

## Key Specifications Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Resolution** | 2048 x 2048 pixels | Full HD format |
| **Pixel Size** | 5.5 x 5.5 µm² | |
| **Optical Format** | 1" (16 mm diagonal) | Compatible with C-mount lenses |
| **Pixel Type** | Global shutter with pinned photodiode | Eliminates motion blur |
| **Fill Factor** | 42% (w/o microlens) | |
| **Quantum Efficiency × Fill Factor** | 60% @ 550 nm (with microlenses) | |
| **Maximum Frame Rate** | 180 fps @ 10-bit, 37.5 fps @ 12-bit | |
| **ADC Resolution** | 10-bit or 12-bit selectable | |

## Optical Performance Parameters

### Sensitivity and Noise
- **Full Well Capacity**: 13.5 ke⁻ (pinned photodiode pixel)
- **Conversion Gain**: 0.075 LSB/e⁻ (10-bit mode, unity gain)
- **Sensitivity**: 5.56 V/lux·s (0.27 A/W with microlenses @ 550 nm)
- **Temporal Noise**: 13 e⁻ (analog domain, read noise)
- **Dynamic Range**: 60 dB
- **Dark Current**: 125 e⁻/s @ 25°C (doubles every 6.5°C increase)

### Fixed Pattern Noise
- **DSNU**: 3 LSB/s (10-bit mode)
- **Fixed Pattern Noise**: <1 LSB RMS (<0.1% of full swing, 10-bit mode)
- **PRNU**: <1% RMS of signal

### Global Shutter Performance
- **Shutter Type**: Pipelined global shutter (exposure during readout)
- **Parasitic Light Sensitivity**: <1/50,000
- **Shutter Efficiency**: >99.998%

## Spectral Response

### Monochrome Version
- **Peak QE**: ~60% @ 550 nm (with microlenses)
- **Spectral Range**: 400-1000 nm (see spectral curves)
- **Cover Glass**: D263 plain or AR-coated
- **Enhanced IR Response**: E12 variant available (doubled QE @ 900 nm: 8% → 16%)

### Color Version (if applicable)
- **Color Pattern**: RGB Bayer filter
- **IR Cutoff**: External IR filter recommended
- **Microlenses**: Always included with color version

## Timing and Control

### Exposure Control
- **Exposure Modes**: Internal or external timing control
- **Minimum Exposure Time**: 
  - Internal mode: 25.8 µs (14.24 µs with fot_length=10)
  - External mode: 23.14 µs (11.58 µs with fot_length=10)
- **Exposure Range**: Programmable from minimum to several seconds

### Frame Timing
- **Frame Rate Calculation**: 
  - FOT (Frame Overhead Time) = 59.125 µs (default)
  - Image readout time = 5.504 ms (full resolution)
  - Total frame time = 5.5631 ms → 180 fps

## Data Interface

### LVDS Outputs
- **Data Channels**: 16 LVDS channels @ 480 Mbps each
- **Additional Channels**: 1 clock + 1 control channel
- **Output Configurations**: 16, 8, 4, or 2 channels selectable
- **Data Format**: 10-bit or 12-bit per pixel

### Control Signals
- **DVAL**: Valid pixel data indicator
- **LVAL**: Valid line data indicator  
- **FVAL**: Valid frame data indicator
- **Training Pattern**: Programmable pattern for receiver synchronization

## Power Requirements

### Supply Voltages
| Supply | Voltage | Usage |
|--------|---------|-------|
| **VDD20** | 2.0-2.15 V | LVDS, ADC |
| **VDD33** | 3.2-3.4 V | Analog supply, PGA |
| **VDDPIX** | 2.9-3.1 V | Pixel array |
| **Vres_h** | 3.2-3.4 V | Pixel reset |

### Power Consumption
- **Total Power**: 550-1200 mW (configuration dependent)
- **Breakdown**: 
  - VDD20: 750-790 mW
  - VDD33: 220-300 mW
  - VDDPIX: 60 mW
  - Vres_h: 50 mW

## Clock Requirements

### Input Clocks
- **CLK_IN**: 5-48 MHz master clock
- **LVDS_CLK**: 50-480 MHz (10× or 12× CLK_IN)
- **SPI_CLK**: Up to 48 MHz

### Clock Relationships
- **10-bit mode**: LVDS_CLK = 10 × CLK_IN
- **12-bit mode**: LVDS_CLK = 12 × CLK_IN

## Environmental Specifications

### Temperature
- **Operating Range**: -30°C to +70°C junction temperature
- **Storage Range**: +20°C to +40°C (30-60% RH non-condensing)
- **Temperature Sensor**: On-chip 16-bit digital sensor

### Reliability
- **ESD Rating**: Class 1A HBM (±2 kV)
- **Package**: Custom ceramic (µPGA, LGA, or LCC)
- **RoHS**: Compliant

## Star Tracker Specific Considerations

### Advantages for Star Tracking
1. **Global Shutter**: Eliminates motion blur during satellite movement
2. **High Sensitivity**: Good for low-light star detection
3. **Low Noise**: 13 e⁻ read noise supports dim star detection
4. **Precise Timing**: Accurate exposure control for consistent measurements
5. **High Dynamic Range**: Can handle bright and dim stars simultaneously

### Key Design Parameters
- **Pixel Scale**: With appropriate optics, determines star position accuracy
- **Exposure Time**: Balance between SNR and star saturation
- **Gain Settings**: 4 analog gain levels (1×, 1.2×, 1.4×, 1.6×) plus ADC gain
- **Windowing**: Up to 8 programmable windows for ROI processing
- **Frame Rate**: High speed capability for rapid attitude updates

### Recommended Settings for Star Tracking
- **Bit Mode**: 12-bit for maximum dynamic range
- **Gain**: Optimize based on star magnitude requirements
- **Exposure**: Typically 10-100 ms depending on platform stability
- **Output Mode**: May use fewer channels to reduce power/data bandwidth

## Programming Interface

### SPI Configuration
- **Interface**: 4-wire SPI up to 48 MHz
- **Registers**: Extensive register set for all operational parameters
- **Key Programmable Features**:
  - Exposure time and mode
  - Gain and offset
  - Windowing parameters
  - Output format and channels
  - Power management

### Control Pins
- **SYS_RES_N**: System reset (active low)
- **FRAME_REQ**: Frame request trigger
- **T_EXP1/T_EXP2**: External exposure control (optional)

## Package Information

### Available Packages
- **µPGA**: 95 pins, through-hole
- **LGA**: 95 pins, surface mount
- **LCC**: 92 pins, leadless chip carrier

### Key Dimensions
- **Die Size**: Approximately 11.3 × 11.3 mm active area
- **Package Size**: ~16 × 16 mm
- **Optical Center**: Precisely specified for alignment

---

*This reference is based on CMV4000 datasheet v7-00. For complete details and latest specifications, refer to the full datasheet from ams-OSRAM.*
