
from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

folder = Path(__file__).resolve().parent

angles = np.array([10, 15, 20, 25, 30, 40, 60])

measurement_files = [
    "run_10_deg_1_foil_oct_5_2026.Chn",
    "run_15_deg_1_foil_oct_5_2026.Chn",
    "run_20_deg_1_foil_oct_5_2026.Chn",
    "run_25_deg_1_foil_oct_7_2026_16_0_500.Chn",
    "run_30_deg_1_foil_oct_13_2026.Chn",
    "run_40_deg_1_foil_oct_8-9_2026_16_0_500.Chn",
    "run_60_deg_1_foil_oct_9-13_2026_combined_excluding_096.Chn"
]

calibration_files = [
    "test_0_deg_0_foil_oct_5_2026.Chn",
    "test_0_deg_0_foil_oct_5_2026.Chn",
    "test_0_deg_0_foil_oct_5_2026.Chn",
    "run_0_deg_0_foil_oct_7_2026_16_0_500.Chn",
    "run_0_deg_0_foil_oct_13_2026_16_0_500.Chn",
    "run_0_deg_0_foil_oct_8_2026_16_0_500.Chn",
    "run_0_deg_0_foil_oct_9_2026_16_0_500.Chn"
]


def read_spectrum(path):
    data = path.read_bytes()
    live_time = struct.unpack_from("<i", data, 12)[0] / 50
    counts = np.frombuffer(data, dtype="<u4", count=2048, offset=32)
    return counts, live_time


# Read measurements and calibrations
counts = []
times = []

for index in range(len(angles)):
    angle = angles[index]
    directory = folder / "runs" / f"{angle}deg"

    measurement_path = directory / measurement_files[index]
    calibration_path = directory / calibration_files[index]

    spectrum, live_time = read_spectrum(measurement_path)
    calibration, _ = read_spectrum(calibration_path)

    # Find the alpha peak
    peak_channel = np.argmax(calibration[900:1800]) + 900

    # Convert 2 MeV to channels
    cutoff_channel = int(np.ceil(peak_channel * 2 / 5.48))

    # Count events above cutoff
    alpha_counts = sum(spectrum[cutoff_channel:])

    counts.append(alpha_counts)
    times.append(live_time)


# Calculate rates and uncertainties
counts = np.array(counts)
times = np.array(times)

rates = counts / times
poisson_errors = np.sqrt(counts) / times


def plum_pudding_model(angle):
    angle = np.asarray(angle, dtype=float)

    theta_m = 1.0
    reference_angle = 10.0
    reference_rate = rates[0]

    return reference_rate * np.exp(
        -(angle**2 - reference_angle**2) / theta_m**2
    )


# Triangular detector response
def g(phi, theta, theta_0):
    if abs(theta - phi) < theta_0:
        return 1 - abs(theta - phi) / theta_0

    return 0


# Rutherford scattering
def rutherford(theta):
    return 1 / np.sin(theta / 2)**4


# Convolved Rutherford scattering
def convolved_rutherford(phi, C_0, theta_0):
    phi = np.deg2rad(phi)
    theta_0 = np.deg2rad(theta_0)

    def integrand(theta):
        return g(phi, theta, theta_0) * rutherford(theta)

    lower = max(0, phi - theta_0)
    upper = min(np.pi, phi + theta_0)

    result, error = quad(
        integrand,
        lower,
        upper,
        points=[phi]
    )

    return C_0 * result


def model(angle, normalization, width=0):
    angle = np.asarray(angle, dtype=float)

    # Ordinary Rutherford scattering
    if width == 0:
        return normalization / np.sin(
            np.deg2rad(angle / 2)
        )**4

    # Convolved Rutherford scattering
    predictions = []

    for current_angle in np.atleast_1d(angle):
        prediction = convolved_rutherford(
            current_angle,
            normalization,
            width
        )

        predictions.append(prediction)

    predictions = np.array(predictions)

    if angle.ndim == 0:
        return predictions[0]

    return predictions


def uncertainties(angle, normalization, width, poisson_errors):
    # Estimate slope at each angle
    rate_above = model(angle + 0.01, normalization, width)
    rate_below = model(angle - 0.01, normalization, width)

    slope = (rate_above - rate_below) / 0.02

    # Combine both uncertainties vertically
    angular_error = slope * 1.0

    return np.sqrt(poisson_errors**2 + angular_error**2)


def chi_squared(parameters, selected, convolved):
    normalization = np.exp(parameters[0])

    width = 0
    if convolved:
        width = parameters[1]

    predicted_rates = model(angles[selected], normalization, width)

    errors = uncertainties(
        angles[selected],
        normalization,
        width,
        poisson_errors[selected]
    )

    total_chi_squared = 0

    for index in range(len(predicted_rates)):
        difference = rates[selected][index] - predicted_rates[index]
        total_chi_squared += (difference / errors[index])**2

    return total_chi_squared


def fit_model(selected, convolved):
    # Choose initial fitting parameters
    initial_parameters = [np.log(0.00007)]
    parameter_bounds = [(-25, -5)]

    if convolved:
        initial_parameters.append(8)
        parameter_bounds.append((0.01, 9.3))

    # Minimize chi-squared
    result = minimize(
        chi_squared,
        initial_parameters,
        args=(selected, convolved),
        bounds=parameter_bounds
    )

    normalization = np.exp(result.x[0])
    width = 0

    if convolved:
        width = result.x[1]

    return normalization, width


# Compare data with plum pudding prediction
plum_predictions = plum_pudding_model(angles)

reference_angle = 10
selected = angles != reference_angle

# Calculate uncertainties
plum_rate_above = plum_pudding_model(angles + 0.01)
plum_rate_below = plum_pudding_model(angles - 0.01)

plum_slope = (plum_rate_above - plum_rate_below) / 0.02
plum_angular_error = plum_slope * 1.0

plum_errors = np.sqrt(
    poisson_errors**2 + plum_angular_error**2
)

# Calculate chi-squared
plum_chi2 = np.sum(
    ((rates[selected] - plum_predictions[selected])
     / plum_errors[selected])**2
)

plum_dof = sum(selected)

print("plum pudding")
print("Theta_m:", 1.0, "degrees")
print("Normalized at:", reference_angle, "degrees")
print("Chi-squared:", plum_chi2)
print("Degrees of freedom:", plum_dof)
print("Reduced chi-squared:", plum_chi2 / plum_dof)
print()


# Plot measured data
plt.figure(figsize=(7, 5))

plt.errorbar(
    angles,
    rates,
    yerr=plum_errors,
    fmt="o",
    color="black",
    capsize=3,
    label="Measured data"
)

# Plot the predicted curve
plot_angles = np.linspace(10, 65, 400)
plot_rates = plum_pudding_model(plot_angles)

plt.plot(
    plot_angles,
    plot_rates,
    "r-",
    label=r"Plum pudding ($\theta_m=1^\circ$)"
)

plt.yscale("log")
plt.ylim(1e-5, 10)

plt.xlabel("Scattering angle (degrees)")
plt.ylabel("Scattering rate (counts/s)")
plt.title("Plum pudding fit")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(folder / "plum_pudding_fit.png", dpi=180)
plt.show()


for convolved in [False, True]:
    if convolved:
        name = "convolved"
    else:
        name = "rutherford"

    # First fit all measurements
    selected = np.ones(len(angles), dtype=bool)
    normalization, width = fit_model(selected, convolved)

    # Identify points beyond 5 sigma
    predictions = model(angles, normalization, width)
    errors = uncertainties(angles, normalization, width, poisson_errors)

    residuals = (rates - predictions) / errors
    selected = abs(residuals) <= 5

    # Refit remaining measurements
    normalization, width = fit_model(selected, convolved)

    errors = uncertainties(angles, normalization, width, poisson_errors)

    parameters = [np.log(normalization)]
    if convolved:
        parameters.append(width)

    chi2 = chi_squared(parameters, selected, convolved)
    degrees_of_freedom = sum(selected) - len(parameters)

    print(name)
    print("Excluded angles:", angles[~selected])
    print("Chi-squared:", chi2)
    print("Degrees of freedom:", degrees_of_freedom)
    print("Reduced chi-squared:", chi2 / degrees_of_freedom)

    if convolved:
        print("Half-width:", width, "degrees")

    print()

    # Plot included and excluded data
    plt.figure(figsize=(7, 5))

    plt.errorbar(
        angles[selected],
        rates[selected],
        yerr=errors[selected],
        fmt="o",
        color="black",
        capsize=3,
        label="Included data"
    )

    if any(~selected):
        plt.errorbar(
            angles[~selected],
            rates[~selected],
            yerr=errors[~selected],
            fmt="o",
            color="gray",
            capsize=3,
            label="Excluded (5σ)"
        )

    # Plot the fitted curve
    plot_angles = np.linspace(10, 65, 400)
    plot_rates = model(plot_angles, normalization, width)

    plt.plot(plot_angles, plot_rates, "r-", label="Best fit")

    plt.yscale("log")
    plt.xlabel("Scattering angle (degrees)")
    plt.ylabel("Scattering rate (counts/s)")
    plt.title(name.capitalize() + " fit")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(folder / f"{name}_5sigma_fit.png", dpi=180)
    plt.show()
