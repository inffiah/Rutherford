from pathlib import Path
import struct
import numpy as np

folder = Path(__file__).resolve().parent

gold_measurement = (
    folder / "runs" / "20deg" /
    "run_20_deg_1_foil_oct_5_2026.Chn"
)

gold_calibration = (
    folder / "runs" / "20deg" /
    "test_0_deg_0_foil_oct_5_2026.Chn"
)

# Find titanium measurement
titanium_folder = folder / "titanium"

titanium_files = [
    path for path in titanium_folder.iterdir()
    if path.is_file()
    and path.suffix.lower() == ".chn"
    and "20_deg" in path.name.lower()
    and "7.62" in path.name
    and "ti_foil" in path.name.lower()
]

if len(titanium_files) != 1:
    raise FileNotFoundError(
        f"Expected one titanium 20-degree file, found {len(titanium_files)}."
    )

titanium_measurement = titanium_files[0]


def read_spectrum(path):
    data = path.read_bytes()

    live_time = struct.unpack_from("<i", data, 12)[0] / 50

    counts = np.frombuffer(
        data,
        dtype="<u4",
        count=2048,
        offset=32
    )

    return counts, live_time


def find_alpha_peak(path):
    counts, _ = read_spectrum(path)

    peak_channel = np.argmax(counts[900:1800]) + 900

    return int(peak_channel)


def calculate_rate(measurement_path, peak_channel):
    spectrum, live_time = read_spectrum(measurement_path)

    cutoff_channel = int(np.ceil(peak_channel * 2 / 5.48))

    alpha_counts = int(np.sum(spectrum[cutoff_channel:]))

    rate = alpha_counts / live_time
    error = np.sqrt(alpha_counts) / live_time

    return (
        rate,
        error,
        alpha_counts,
        live_time,
        cutoff_channel
    )


# Check gold files
for path in [gold_measurement, gold_calibration]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")


# Gold calibration
gold_peak_channel = find_alpha_peak(gold_calibration)


# Find 0-degree no-foil calibration files from other days
calibration_files = [
    path
    for path in (folder / "runs").rglob("*")
    if path.is_file()
    and path.suffix.lower() == ".chn"
    and "0_deg_0_foil" in path.name.lower()
    and "oct_14" not in path.name.lower()
]

if not calibration_files:
    raise FileNotFoundError(
        "No 0-degree no-foil calibration files found."
    )


# Average alpha peak channels
peak_channels = []

print("\nCalibration measurements from other days:")

for path in sorted(calibration_files):
    peak = find_alpha_peak(path)
    peak_channels.append(peak)

    print(f"{path.name}: channel {peak}")

titanium_peak_channel = int(round(np.mean(peak_channels)))

print(f"\nAverage alpha peak channel: {titanium_peak_channel}")
print("Note: This is an approximate titanium calibration.")
print("Different amplifier gains can shift the peak channel.")


# Calculate gold scattering rate
gold_rate, gold_error, gold_counts, gold_time, gold_cutoff = (
    calculate_rate(
        gold_measurement,
        gold_peak_channel
    )
)


# Calculate titanium scattering rate
titanium_rate, titanium_error, titanium_counts, titanium_time, titanium_cutoff = (
    calculate_rate(
        titanium_measurement,
        titanium_peak_channel
    )
)


# Measured Au/Ti ratio
if gold_rate <= 0 or titanium_rate <= 0:
    raise ValueError("Cannot calculate ratio with zero scattering rate.")

measured_ratio = gold_rate / titanium_rate

ratio_error = measured_ratio * np.sqrt(
    (gold_error / gold_rate)**2 +
    (titanium_error / titanium_rate)**2
)


# Rutherford Z^2 prediction
z_gold = 79
z_titanium = 22

predicted_z2_ratio = (z_gold / z_titanium)**2


# Foil thicknesses
gold_thickness = 1.73       # micrometers, Taohan's calculation
titanium_thickness = 7.62   # micrometers, nominal

gold_density = 19.32        # g/cm^3
titanium_density = 4.54     # g/cm^3

gold_molar_mass = 196.97    # g/mol
titanium_molar_mass = 47.867  # g/mol

gold_thickness_cm = gold_thickness * 1e-4
titanium_thickness_cm = titanium_thickness * 1e-4

avogadro = 6.02214076e23


# Target atoms per unit area
gold_areal_density = (
    gold_density * gold_thickness_cm
    / gold_molar_mass * avogadro
)

titanium_areal_density = (
    titanium_density * titanium_thickness_cm
    / titanium_molar_mass * avogadro
)


# Thickness-corrected Rutherford prediction
predicted_corrected_ratio = (
    gold_areal_density / titanium_areal_density
) * predicted_z2_ratio


# Compare measurement and prediction
percent_difference = (
    abs(measured_ratio - predicted_corrected_ratio)
    / predicted_corrected_ratio * 100
)

difference_sigma = (
    abs(measured_ratio - predicted_corrected_ratio)
    / ratio_error
)


# Print results
print("\n20-degree Z dependence comparison\n")

print("Gold:")
print(f"Calibration peak channel: {gold_peak_channel}")
print(f"2 MeV cutoff channel: {gold_cutoff}")
print(f"Counts: {gold_counts}")
print(f"Live time: {gold_time:.2f} s")
print(f"Rate: {gold_rate:.5f} +/- {gold_error:.5f} counts/s\n")

print("Titanium:")
print(f"Measurement file: {titanium_measurement.name}")
print(f"Average calibration peak channel: {titanium_peak_channel}")
print(f"2 MeV cutoff channel: {titanium_cutoff}")
print(f"Counts: {titanium_counts}")
print(f"Live time: {titanium_time:.2f} s")
print(f"Rate: {titanium_rate:.5f} +/- {titanium_error:.5f} counts/s\n")

print("Foil thicknesses:")
print(f"Gold: {gold_thickness} micrometers")
print(f"Titanium: {titanium_thickness} micrometers\n")

print("Results:")
print(f"Measured Au/Ti ratio: {measured_ratio:.3f} +/- {ratio_error:.3f}")
print(f"Uncorrected Z^2 prediction: {predicted_z2_ratio:.3f}")
print(f"Thickness-corrected prediction: {predicted_corrected_ratio:.3f}")
print(f"Percent difference: {percent_difference:.1f}%")
print(f"Difference in counting sigma: {difference_sigma:.2f}")

print("\nWARNING: Titanium calibration is approximate.")
print("The average uses no-foil peaks from other days.")
print("This is only reliable if amplifier settings are comparable.")
print("The comparison also assumes comparable incident flux and detector conditions.")