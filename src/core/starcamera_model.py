import matplotlib
import numpy as np

# Constants
h = 6.626e-34  # planck's constant, Joule-s
c = 2.997e8  # speed of light, m/s
solar_flux_constant = 1367.5  # solar flux at earth, w/m^2
solar_magnitude = -26.7  # apparent magnitude of sun
airy_disc_pwratio = 0.838  # fraction of energy in the airy disc

# tighe: 0.5 valid for gaussian - need to verify for airy
FWHM_pwratio = 0.5  # fraction of energy in FWHM, depends on the PSF shape
FWHM_ratio = 2.39  # airy disc diameter to FWHM diameter
visualband = [0.400, 0.800]  # visual bandpass, um


# star is a model of a star's properties and its flux calculation.
#  magnitude: apparent magnitude of the star
#  passband: a list describing the min and max wavelengths in micrometers (um)
#  cent_wavelength: calculated central wavelength of the passband
#  flux: calculated photon flux (photons/s/m^2)
class star:
    def __init__(self, magnitude, passband):
        self.magnitude = magnitude  # apparent magnitude, Mv
        self.passband = passband  # [min wavelength, max wavelength], um
        self.cent_wavelength = (passband[0] + passband[1]) / 2  # central wavelength, um
        self.flux = self.calculate_flux()  # photons/s/m^2

    def calculate_flux(self):
        # Calculate the approximate flux of a M0 star
        M0_flux_const = (solar_flux_constant * 1e-6) / (
            2.5 ** abs(solar_magnitude)
        )  # W/m^2
        # print("M0 flux const: ", str(M0_flux_const), "W/m^2")

        # Calculate the energy of a photon at the central wavelength
        ph_energy = (h * c) / (self.cent_wavelength * 1e-6)  # Joules

        # Calculate the photon flux
        ph_flux = (
            M0_flux_const * (2.5 ** (-1 * self.magnitude)) / ph_energy
        )  # photons/s/m^2

        return ph_flux


# scene is a model of the observation scene parameters.
#  int_time: integration time in milliseconds (ms)
#  temp: temperature in degrees Celsius (°C)
#  slew_rate: slew rate in degrees per second (°/s)
#  fwhm: full width at half maximum in pixels (px), default is 1
class scene:
    def __init__(self, int_time, temp, slew_rate, fwhm=1):
        self.int_time = int_time  # integration time, ms
        self.temp = temp  # temperature, °C
        self.slew_rate = slew_rate  # slew rate, °/s
        self.fwhm = fwhm  # full width at half maximum, px


# star camera is a hw model of a star tracker. it contains two components and parameters obtained via the properties of either
#  optic: an optical_stack object that contains the properties of the optical system
#  fpa: a "fpa" object that defines the properties of the focal plane array of the star camera
#  passband: a 1D vector describing the min and max operating wavelengths in (um) of the star camera
#  cent_wavelength: a calculated value of the central wavelength of the passband in (um)
#  psf_multiplier: a scalar value [1, inf] used to reduce the performance of the optic below the diffraction limited airy disk
#  f_length: focal length of the star camera in (mm), defined by the FFOV of the optic and the physical dimension of the fpa
#  aperature: the diameter of the aperture in  (mm) based on the f-stop and focal length
#  aperature_area: the area of the aperture in (mm^2), calculated as pi * (aperature/2)^2
class star_camera:
    def __init__(self, optic, fpa, passband, psf_mult=1):
        self.optic = optic  # optic_stack object containing optical system properties
        self.fpa = fpa  # fpa object defining the focal plane array properties
        self.passband = passband  # [min wavelength, max wavelength], um
        self.cent_wavelength = (passband[0] + passband[1]) / 2  # central wavelength, um

        # psf_mult is derived from simulation of the optic performance, and is what we use to increase the spot size beyond the diffraction limit of the optic
        # for ADR, take the average FWHM of the PSFs from Adam's simulation of the demo optic and feed that into set_fwhm to get the corresponding psf_mult
        self.psf_mult = psf_mult  # PSF multiplier, scalar value [1, inf]

        self.calculate_optic_system()

    # optic_stack is a model of the optical system of the star camera.
    class optic_stack:
        def __init__(self, f_stop, ffov, transmission):
            self.f_stop = f_stop  # f-stop of the optical system
            self.ffov = ffov  # full field of view, degrees
            self.fov_halfang = ffov / 2  # half of the field of view angle, degrees
            self.transmission = (
                transmission  # transmission efficiency of the optical system
            )

    # fpa is a model of the focal plane array of the star camera.
    class fpa:
        def __init__(
            self,
            x_pixels,
            y_pixels,
            pitch,
            qe,
            dark_current_ref,
            dark_current_ref_temp,
            dark_current_coefficient,
            full_well,
            read_noise,
            bit_depth,
        ):
            self.x_pixels = x_pixels  # width in pixels
            self.y_pixels = y_pixels  # height in pixels
            self.pitch = pitch  # width of pixel in um
            self.qe = qe  # quantum efficiency, fraction
            self.dark_current_ref = dark_current_ref  # dark current reference, e-/s
            self.dark_current_ref_temp = (
                dark_current_ref_temp  # dark current reference temperature, °C
            )
            self.dark_current_coefficient = dark_current_coefficient  # dark current temperature coefficient, e-/s/°C
            self.full_well = full_well  # full well capacity, e-
            self.read_noise = read_noise  # read noise, e-
            self.bit_depth = bit_depth  # bit depth of the ADC
            self.sensitivity = (
                self.calculate_sensitivity()
            )  # sensitivity, e-/ADU or e-/LSB
            self.quant_noise = self.calculate_quant_noise()  # quantization noise, e-
            self.x_width = self.x_pixels * self.pitch / 1000  # width in mm
            self.y_width = self.y_pixels * self.pitch / 1000  # height in mm

        # Calculate the sensitivity of the FPA
        def calculate_sensitivity(self):
            sensitivity = self.full_well / 2**self.bit_depth  # e-/ADU
            return sensitivity

        # Calculate the quantization noise of the FPA
        def calculate_quant_noise(self):
            quant_noise = self.sensitivity / np.sqrt(12)  # RMS e-
            return quant_noise

        # Calculate the dark signal noise of the FPA
        def calculate_dark_signal(
            self, temperature, int_time
        ):  # temperature in °C, int_time in ms
            # Calculate the dark signal noise
            # tighe: this is according the the CMV4000 datasheet, which does not seem to match the literature which expects an exp relationship between dark current and temperature
            self.dark_signal = (
                self.dark_current_ref
                * 2
                ** (
                    (temperature - self.dark_current_ref_temp)
                    / self.dark_current_coefficient
                )
                * (int_time / 1000)
            )  # e-

            # Calculate the shot noise of the dark signal
            # tighe: not clear to me that dark signal noise and dark signal shot noise are different things - do you have a reference for these calcs?
            self.dark_noise = np.sqrt(self.dark_signal)  # e-

            return

    # Calculate the PSF multiplier based on the desired FWHM
    def set_fwhm(self, fwhm):
        self.psf_multi = (FWHM_ratio * fwhm) / self.airy_disc  # dimensionless
        return

    # Calculate the optical system parameters
    def calculate_optic_system(self):
        # Calculate the focal length based on the FOV and FPA dimensions
        self.f_length = (max(self.fpa.x_width, self.fpa.y_width) / 2) / np.tan(
            np.radians(self.optic.fov_halfang)
        )  # mm

        # Calculate the aperture diameter based on the focal length and f-stop
        self.aperature = self.f_length / self.optic.f_stop  # mm

        # Calculate the aperture area
        self.aperature_area = np.pi * (self.aperature / 2) ** 2  # mm^2

        # Calculate the Airy disk diameter
        self.airy_disc = (
            2.44 * self.cent_wavelength * self.optic.f_stop / self.fpa.pitch
        )  # px

        # Calculate the PSF diameter
        self.psf_d = self.airy_disc * self.psf_mult  # px

        # Calculate the pixel angular resolution
        self.pixel_angular_resolution = (
            np.degrees(np.arctan((self.fpa.pitch * 1e-3) / self.f_length)) * 3600
        )  # arcsec

        return


# Global functions

# calculate_optical_signal calculates the optical signal received by the camera from a star.
# calculate_well_charge calculates the average per pixel well charge.
# calculate_signal calculates the signal, noise, and other parameters for different observation modes.
# pixel_blur calculates the pixel blur assuming the PSF starts centered on the pixel.
# calculate_pixel_noise calculates the noise in a pixel.
# center_spot calculates the number of lit pixels for a given PSF diameter, assuming the PSF is centered on a single pixel.
# dynamic_threshold calculates the dynamic threshold for signal detection.
# print_class_variables recursively prints all variables in a class, including nested classes.


# Calculate the optical signal received by the camera from a star
def calculate_optical_signal(star, camera, scene):
    sig = (
        star.flux
        * camera.aperature_area
        * camera.optic.transmission
        * (scene.int_time / 1000)
    )  # photons
    return sig


# Calculate average well charge per-pixel
def calculate_well_charge(ph_signal, camera):
    # tighe: i think this might be missing the fill factor of the pixel. the CMV4000 spec provides a QE*FF = 60% with microlenses, or a FF of 42% without
    well_charge = ph_signal * camera.fpa.qe  # e-

    # Limit the well charge to the full well capacity
    if well_charge > camera.fpa.full_well:
        well_charge = camera.fpa.full_well  # e-

    return well_charge


# Calculate the signal for different observation modes
def calculate_signal(mode, star, camera, scene):
    # tighe: i would make this match-case function for simplicity

    # Single pixel PSF, no blur
    # - Adjusts the integration time to limit pixel blur to 0.5px
    # - Assumes PSF falls on a single pixel
    if mode == "single_px_no_blur":
        # Saves previous int time
        previous_time = scene.int_time

        # Calculate time for point source at max slew to traverse 0.5px
        scene.int_time = (0.5 * camera.pixel_angular_resolution) / (
            (scene.slew_rate / 1000) * 3600
        )  # ms

        # Calculate signal for that integration time, scale by airy disc power ratio
        ph_sig = (
            calculate_optical_signal(star, camera, scene) * airy_disc_pwratio
        )  # photons

        # Restore previous int time
        scene.int_time = previous_time

        pixels_covered = 1  # pixels

    # tighe: the pixels_covered calculation in the following modes is tricky and gets at 'what do we mean by SNR'. this approach is conservative, but potentially too convservative considering that we will be thresholding. for an actual PSF, SNR will be a function of location - so for this mode, do we want SNR of the brightest pixel, or the average SNR across all pixels? we probably want to be able to plot both and show a min/max SNR to help us set thresholds

    # Static PSF, no blur
    # - Produces the average per-pixel signal for non-moving, defocused psf
    if mode == "fwhm_static":
        camera.set_fwhm(scene.fwhm)
        pixels_covered = scene.fwhm ** 2  # pixels
        ph_sig = (
            calculate_optical_signal(star, camera, scene) * airy_disc_pwratio
        ) / pixels_covered  # photons/pixel

    # Dynamic PSF, blurred
    # - Produces the average per-pixel signal for moving defocused PSF
    if mode == "psf_moving":
        camera.setfwhm(scene.fwhm)
        pixels_covered = (np.ceil(camera.psf_d) ** 2) + (
            camera.psf_d * pixel_blur(camera, scene)
        )  # pixels
        ph_sig = (
            calculate_optical_signal(star, camera, scene) * airy_disc_pwratio
        ) / pixels_covered  # photons/pixel

    # Dynamic FWHM, blurred
    # - Produces the average per-pixel signal for moving defocused PSF
    # - Simulates thresholding by limiting to the FWHM of the PSF
    if mode == "psf_FWHMonly":
        camera.set_fwhm(scene.fwhm)  # set FWHM
        width = center_spot(scene.fwhm)  # pixels
        pixels_covered = (np.ceil(width) ** 2) + (
            width * pixel_blur(camera, scene)
        )  # pixels
        total_photons = calculate_optical_signal(star, camera, scene)
        ph_sig = (
            total_photons * FWHM_pwratio
        ) / pixels_covered  # photons/pixel
        print("n photons: " + str(total_photons))

    # Calculate remaining parameters
    well_charge = calculate_well_charge(ph_sig, camera)  # e-

    # If we can perfectly calibrate out the dark signal, then it doesn't factor into SNR (ie. we will remove the DC bias). if we cannot calibrate it perfectly, then it starts to matter a lot more
    noise = calculate_pixel_noise(ph_sig, camera, scene)  # e-

    # This is an upper bound on the number of electrons we would capture in a pixel, the lower bound being -noise instead of +noise
    # original
    # signal = well_charge + noise + camera.fpa.dark_signal  # e-
    # updated (worst case signal)
    signal = well_charge - noise + camera.fpa.dark_signal  # e-

    # dig_signal = np.round(signal / camera.fpa.sensitivity)  # ADC
    snr = well_charge / noise  # dimensionless
    saturation = well_charge / camera.fpa.full_well  # fraction

    return snr, saturation, signal, noise, pixels_covered, well_charge


# Calculate pixel blur assuming that the PSF starts centered on the pixel
def pixel_blur(camera, scene):
    # Calculate angular displacement in arcseconds
    ang_displacement = (scene.slew_rate / 1000) * 3600 * scene.int_time  # arcsec

    # Convert angular displacement to pixel displacement
    pixels = ang_displacement / camera.pixel_angular_resolution  # px

    # Calculate the pixel displacement, rounding up to the nearest integer
    pixel_displacement = np.ceil(pixels - 0.5)  # px

    return pixel_displacement


# Simple CMOS APS noise model with shot noise, dark signal noise, dark signal shot noise, quantization noise, and read noise
# - sig: Signal in electrons (e-)
def calculate_pixel_noise(sig, camera, scene):
    # Calculate dark signal noise and dark signal shot noise
    camera.fpa.calculate_dark_signal(scene.temp, scene.int_time)
    dark_noise = camera.fpa.dark_noise  # e-

    # Calculate quantization noise
    quant_noise = camera.fpa.calculate_quant_noise()  # e-

    # Calculate shot noise from the signal
    shot_noise = np.sqrt(camera.fpa.qe * sig)  # e-

    # Read noise from the FPA
    read_noise = camera.fpa.read_noise  # e-

    # Combine all noise components in quadrature
    pixel_noise = np.sqrt(
        np.sum(np.square([dark_noise, quant_noise, shot_noise, read_noise]))
    )  # e-
    return pixel_noise


# Calculate the width in pixels of a square filter given the input diameter of the spot
# - Assume spot is circular, and the width is calculated as the smallest odd integer greater than or equal to the diameter
def center_spot(diameter):
    # Calculate the width of the square filter
    width = 2 * np.ceil(diameter / 2 - 0.5) + 1  # pixels

    # Return the calculated width
    return width


# Calculate the dynamic threshold for signal detection
# - Assume signal and noise are distributed across a region, and the threshold is calculated based on a k-sigma margin.
def dynamic_threshold(signal, noise, region, k, camera, scene, numpx):
    # Calculate the total signal from the number of lit pixels
    # min signal = well_charge - noise + camera.fpa.dark_signal

    lit_px_signal = numpx * signal  # e-

    # Calculate the noise for unlit pixels
    unlit_px_noise = calculate_pixel_noise(0, camera, scene)  # e-


    # tighe: i think that what you want here is actually the dark signal, not just the dark signal noise
    # Calculate the total dark signal for unlit pixels
    # original
    # unlit_px_signal = (region**2 - numpx) * unlit_px_noise  # e-
    # updated
    unlit_px_signal = (region**2 - numpx) * (
        camera.fpa.dark_signal + unlit_px_noise
    )  # e-

    # Calculate the mean signal across the region
    mean = (lit_px_signal + unlit_px_signal) / region**2  # e-

    # tighe: for readability i would have calculated the variance first then the stddev as a separate line. easier to debug but this checks out
    # Calculate the standard deviation (sigma) of the signal distribution
    # original
    sigma = np.sqrt(
        (
            (signal - mean) ** 2 * numpx
            + (unlit_px_noise - mean) ** 2 * (region**2 - numpx)
        )
        / region**2
    )  # e-
    # updated
    # Calculate the variance of the signal distribution



    # variance = (numpx * (signal - mean)) ** 2  # signal weighted by number of lit pixels
    #    + (region**2 - numpx) * (unlit_px_signal - mean) ** 2

    # Calculate the standard deviation (sigma) as the square root of the variance
    # sigma = np.sqrt(variance)  # e-

    # Calculate the threshold based on the mean and k-sigma margin
    threshold = mean + k * sigma  # e-

    # Calculate the margin of the signal above the threshold
    margin = (signal - threshold) / sigma  # dimensionless

    return margin, threshold
