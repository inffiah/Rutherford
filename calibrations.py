from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
runs = folder / "runs"

angles = [10, 15, 20, 25, 30, 40, 60]

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


def read_spectrum(filename):
    content = Path(filename).read_bytes()
    live_time = struct.unpack_from("<i", content, 12)[0] / 50
    counts = np.frombuffer(content, dtype="<u4", count=2048, offset=32).copy()
    return counts, live_time


def calibration_peak(counts):
    return np.argmax(counts[900:1800]) + 900


plt.figure(figsize=(8, 5))

calibration_angles = [10, 25, 40, 60, 30]
calibration_days = [5, 7, 8, 9, 13]

for index in range(len(calibration_angles)):
    angle = calibration_angles[index]
    day = calibration_days[index]

    calibration_index = angles.index(angle)
    directory = runs / f"{angle}deg"

    spectrum, _ = read_spectrum(
        directory / calibration_files[calibration_index]
    )

    peak = calibration_peak(spectrum)
    energies = np.arange(len(spectrum)) * 5.48 / peak

    normalized_counts = spectrum / spectrum.max()

    plt.plot(
        energies,
        normalized_counts,
        label=f"Oct {day}, peak channel {peak}"
    )

plt.axvline(2, color="red", linestyle="--", label="2 MeV cutoff")
plt.xlabel("Energy (MeV)", labelpad=10)
plt.ylabel("Counts / peak counts", labelpad=10)
plt.title("Zero-Foil Energy Calibrations", pad=15)
plt.xlim(0, 7)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(folder / "energy_calibrations.png", dpi=180)
plt.close()


fig, axes = plt.subplots(2, 4, figsize=(18, 8))

bin_size = 12

for index in range(len(angles)):
    angle = angles[index]
    directory = runs / f"{angle}deg"

    spectrum, _ = read_spectrum(
        directory / measurement_files[index]
    )

    calibration, _ = read_spectrum(
        directory / calibration_files[index]
    )

    peak = calibration_peak(calibration)
    energies = np.arange(len(spectrum)) * 5.48 / peak

    binned_energies = []
    binned_counts = []

    for start in range(0, len(spectrum), bin_size):
        end = start + bin_size

        binned_energies.append(np.mean(energies[start:end]))
        binned_counts.append(np.sum(spectrum[start:end]))

    row = index // 4
    column = index % 4
    ax = axes[row, column]

    ax.step(
        binned_energies,
        binned_counts,
        where="mid",
        color="steelblue",
        linewidth=1.5
    )

    ax.axvline(
        2,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="2 MeV cutoff"
    )

    ax.set_title(f"{angle}° Scattering", pad=14)
    ax.set_xlim(0, 7)
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)", labelpad=10)
    ax.set_ylabel("Counts per bin", labelpad=10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

axes[1, 3].axis("off")

fig.suptitle("Gold-Foil Energy Spectra", fontsize=16, y=0.99)

fig.subplots_adjust(
    left=0.06,
    right=0.99,
    bottom=0.10,
    top=0.89,
    hspace=0.48,
    wspace=0.38
)

fig.savefig(folder / "gold_foil_energy_spectra.png", dpi=200)
plt.show()

print("Saved energy_calibrations.png")
print("Saved gold_foil_energy_spectra.png")