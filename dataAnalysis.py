#%%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#%%
# Insert the input power levels corresponding to each acquisition file (2.dat to last.dat)
# op_powers_1_perc_output_coupler = np.array([])  #MEASURED VALUES

# input_powers = op_powers_1_perc_output_coupler/0.0091 #poer at the output of the 99% coupler output, entering the detector




# ==============================================================================
# 1. SETUP PATH & PARAMETER CONFIGURATION
# ==============================================================================
folder_path = r"C:\Users\cater\pyrpl_user_dir\curve\first_test_on_neg_PD"
start_file = 1
end_file = 5  

# Unified Welch configurations to avoid array shape mismatches
n_samples_per_segment = 512  
overlap_samples = n_samples_per_segment // 2

# Global dark mode style for clean, uniform aesthetics
plt.style.use('dark_background')

# Tracking variables for the final linear regression
file_indices = []
mean_clearances = []

# ==============================================================================
# 2. EXTRACT BASELINE NOISE FLOOR (1.dat)
# ==============================================================================
noise_file_path = os.path.join(folder_path, f"{start_file}.dat")
if not os.path.exists(noise_file_path):
    raise FileNotFoundError(f"Baseline noise file '{start_file}.dat' missing.")

with open(noise_file_path, "rb") as f:
    noise_obj = pickle.load(f)

noise_time = np.array(noise_obj[2][0])
noise_voltage = np.array(noise_obj[2][1])
fs_noise = 1.0 / (noise_time[1] - noise_time[0])

frequencies_noise, noise_psd = signal.welch(
    noise_voltage, fs=fs_noise, nperseg=n_samples_per_segment, noverlap=overlap_samples
)

# --------------------------------------------------------------------------
# IMAGE ELECTRICAL NOISE
# --------------------------------------------------------------------------
fig_noise, (ax_noise_time, ax_noise_psd) = plt.subplots(
    1, 2, figsize=(14, 5), num="Baseline Electrical Noise (1.dat)"
)

# Time Domain Subplot (Noise)
ax_noise_time.plot(noise_time * 1000, noise_voltage, color='#00ff00', linewidth=0.5)
ax_noise_time.set_xlabel("Time (ms)")
ax_noise_time.set_ylabel("Voltage (V)")
ax_noise_time.set_title("Time Domain: 1.dat (Noise Floor)")
ax_noise_time.grid(True, color='#222222', linestyle='--')

# PSD Subplot (Noise)
ax_noise_psd.semilogy(frequencies_noise / 1e3, noise_psd, color='#00bfff', linewidth=1)
ax_noise_psd.set_xlabel("Frequency (kHz)")
ax_noise_psd.set_ylabel("PSD ($V^2 / Hz$)")
ax_noise_psd.set_title("Raw PSD: 1.dat (Noise Floor)")
ax_noise_psd.grid(True, which='both', color='#222222', linestyle='--')
fig_noise.tight_layout()

# ==============================================================================
# 3. SETUP CONSOLIDATED FIGURES
# ==============================================================================
# Pre-initialize figures for Phases 2, 3, and 4
fig_all_psd = plt.figure(num="Consolidated Raw PSDs", figsize=(11, 6))
fig_avg_clearance = plt.figure(num="Average Clearance Levels", figsize=(11, 6))

# Define colormaps for clean color coordination across multi-line plots
colors = plt.cm.plasma(np.linspace(0.3, 0.9, end_file - start_file))
color_idx = 0


# ==============================================================================
# 4. PROCESSING PIPELINE LOOP
# ==============================================================================
print("Starting execution pipeline...\n")

for i in range(start_file + 1, end_file + 1):
    file_name = f"{i}.dat"
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        continue
        
    print(f"--> Processing: {file_name}")
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    
    time_data = np.array(obj[2][0])
    voltage_data = np.array(obj[2][1])
    fs = 1.0 / (time_data[1] - time_data[0])
    
    # Calculate current file's PSD
    frequencies, current_psd = signal.welch(
        voltage_data, fs=fs, nperseg=n_samples_per_segment, noverlap=overlap_samples
    )
    
    # Calculate clearance metrics
    clearance_db = 10 * np.log10(current_psd / noise_psd)
    mean_val = np.mean(clearance_db)
    
    # Save statistics for the linear fit phase
    file_indices.append(i)
    mean_clearances.append(mean_val)
    
    # --------------------------------------------------------------------------
    # IMAGE 1 PATTERN: Dedicated Subplot Figure per .dat File
    # --------------------------------------------------------------------------
    fig_single, (ax_time, ax_psd_single) = plt.subplots(
        1, 2, figsize=(14, 5), num=f"Acquisition Breakdown: {file_name}"
    )
    
    # Time Domain Subplot
    ax_time.plot(time_data * 1000, voltage_data, color='#00ff00', linewidth=0.5)
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel("Voltage (V)")
    ax_time.set_title(f"Time Domain: {file_name}")
    ax_time.grid(True, color='#222222', linestyle='--')
    
    # PSD Subplot
    ax_psd_single.semilogy(frequencies / 1e3, current_psd, color='#00bfff', linewidth=1)
    ax_psd_single.set_xlabel("Frequency (kHz)")
    ax_psd_single.set_ylabel("PSD ($V^2 / Hz$)")
    ax_psd_single.set_title(f"Raw PSD: {file_name}")
    ax_psd_single.grid(True, which='both', color='#222222', linestyle='--')
    fig_single.tight_layout()
    
    # --------------------------------------------------------------------------
    # IMAGE 2 PATTERN: Consolidated Raw PSD Layout
    # --------------------------------------------------------------------------
    plt.figure(fig_all_psd.number)
    plt.semilogy(frequencies / 1e3, current_psd, alpha=0.7, linewidth=1, label=f"File {i}")

    # --------------------------------------------------------------------------
    # IMAGE 3 PATTERN: Average Clearance Flat Step Layout
    # --------------------------------------------------------------------------
    plt.figure(fig_avg_clearance.number)
    plt.axhline(
        y=mean_val, 
        color=colors[color_idx], 
        linestyle='-', 
        linewidth=2, 
        label=f"File {i} Avg ({mean_val:.2f} dB)"
    )
    
    color_idx += 1

# ==============================================================================
# 5. FINALIZE CONSOLIDATED PLOTS & RENDER LINEAR TREND
# ==============================================================================
# Clean up Consolidated PSD plot (Image 2 Pattern)
plt.figure(fig_all_psd.number)
plt.semilogy(frequencies_noise / 1e3, noise_psd, color='white', linestyle='--', linewidth=1.2, label="Noise Floor (1.dat)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("PSD ($V^2 / Hz$)")
plt.title("Consolidated Power Spectral Density (5 Traces)", fontweight='bold')
plt.grid(True, which='both', color='#222222', linestyle='--')
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
plt.tight_layout()

# Clean up Average Spectral Clearance plot (Image 3 Pattern)
plt.figure(fig_avg_clearance.number)
plt.axhline(0, color='white', linestyle=':', alpha=0.6, label="Noise Floor (0 dB)")
plt.xlim(0, (fs / 2) / 1e3)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Average Clearance (dB)")
plt.title("Clearance Level per Acquisition", fontweight='bold')
plt.grid(True, color='#222222', linestyle='--')
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
plt.tight_layout()

# --------------------------------------------------------------------------
# LINEAR FITTING: Trend Analysis Presentation
# --------------------------------------------------------------------------
#x_data = input_powers
x_data = np.array(file_indices)

y_data = np.array(mean_clearances)

# Compute 1st-degree polynomial fit line (y = mx + c)
slope, intercept = np.polyfit(x_data, y_data, 1)
x_fit = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 100)
y_fit = slope * x_fit + intercept

print("\n--- Linear Regression Statistics ---")
print(f"Calculated Fit Line Equation: Clearance = {slope:.3f} * (File Index) + ({intercept:.3f})")

fig_linear = plt.figure(num="Clearance Linear Fitting", figsize=(8, 5))
plt.scatter(x_data, y_data, color='#00ff00', s=100, zorder=5, label="Calculated Averages")
plt.plot(x_fit, y_fit, color='#ff007f', linestyle='--', linewidth=2, label=f"Linear Fit (Slope: {slope:.2f} dB/file)")

# Attach numeric labels on top of data points
for x, y in zip(x_data, y_data):
    plt.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='white')

plt.xlabel("File Index (Acquisition Number)")
plt.ylabel("Mean Clearance (dB)")
plt.title("Clearance Linearity Fitting Analysis", fontweight='bold')
plt.xticks(x_data)
plt.grid(True, color='#222222', linestyle='--')
plt.legend(loc="lower right")
plt.tight_layout()

print("\nExecution complete! Displaying all generated figures.")
plt.show()
# %%
