# Panos Lelakis, 1083712, Headphone EQ, task 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.ticker import ScalarFormatter

# Load headphone responses
hpfilesnum = 7
hp = []
hp_fs = []
for i in range(1, hpfilesnum + 1):
    fs, data = wavfile.read(f'headphone_responses/hp0{i}.wav')
    hp.append(data)
    hp_fs.append(fs)

# Load targets
targetsnum = 2
target = []
target_fs = []
fs, data = wavfile.read('targets/flat_target.wav')
target.append(data)
target_fs.append(fs)
fs, data = wavfile.read('targets/harman_target.wav')
target.append(data)
target_fs.append(fs)

# Scaling factor in dB for correct plots (further scaling)
scaling_db = 0

# Plot magnitude responses
for i in range(hpfilesnum):
    # Calculate frequency response for headphone
    hp_fft = np.fft.rfft(hp[i], len(hp[i]), norm='forward') # norm='forward' for 1/n scaling
    hp_mag = np.abs(hp_fft)/2 # Divide by 2 for further scaling
    hp_freq = np.linspace(0, hp_fs[i]/2, len(hp_mag))
    hp_mag_db = 20 * np.log10(hp_mag) - scaling_db

    plt.figure(i + 1)

    # Plot headphone response
    plt.subplot(targetsnum + 1, 1, 1)
    plt.plot(hp_freq, hp_mag_db, color='b')
    plt.gca().set_xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().set_yticks([-80, -50, -20, 0, 20, 40])
    plt.xlim([20, 20000])
    plt.ylim([-80, 40])
    plt.title(f'Headphone Response {i + 1}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.gca().grid(which='major', linestyle='--', linewidth=0.7, color='tab:brown')
    plt.gca().grid(which='minor', linestyle=':', linewidth=0.5, color='tab:gray')
    plt.gca().get_ygridlines()[3].set_color('k')
    plt.gca().get_ygridlines()[3].set_linewidth(1)
    plt.gca().minorticks_on()
    # Source for changing color and linewidth for specific gridlines:
    # https://stackoverflow.com/questions/32073616/matplotlib-change-color-of-individual-grid-lines
    # Source for making all ticks visible using ScalarFormatter():
    # https://www.reddit.com/r/learnpython/comments/epwteg/matplotlib_ticklabels_disappearing_in_log_scale/
    
    # Plot target responses in frequency domain
    for j in range(targetsnum):
        # Calculate frequency response of target
        target_fft = np.fft.rfft(target[j], len(hp[i]), norm='forward')
        target_mag = np.abs(target_fft)/2
        target_freq = np.linspace(0, target_fs[j]/2, len(target_mag))
        target_mag_db = 20 * np.log10(target_mag) - scaling_db
        
        # Calculate MSE and RMSE
        mse = np.mean((hp_mag - target_mag) ** 2)
        rmse = np.sqrt(mse)
        # Source for mse and rmse formulas:
        # https://en.wikipedia.org/w/index.php?title=Mean_squared_error&oldid=1207422018

        # Plot headphone response vs target response
        plt.subplot(targetsnum + 1, 1, j + 2)
        plt.plot(hp_freq, hp_mag_db, color='b')
        plt.plot(target_freq, target_mag_db, color='r')
        plt.gca().set_xscale('log')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.gca().set_yticks([-80, -50, -20, 0, 20, 40])
        plt.xlim([20, 20000])
        plt.ylim([-80, 40])
        legend = 'Flat Target' if j == 0 else 'Harman Target'
        plt.title(f'Headphone Response {i + 1} vs {legend} Response')
        plt.legend(['Headphone Response', legend], loc='lower right')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.gca().grid(which='major', linestyle='--', linewidth=0.7, color='tab:brown')
        plt.gca().grid(which='minor', linestyle=':', linewidth=0.5, color='tab:gray')
        plt.gca().get_ygridlines()[3].set_color('k')
        plt.gca().get_ygridlines()[3].set_linewidth(1)
        plt.gca().minorticks_on()
        # Display MSE and RMSE inside the plot
        plt.text(0.03, 0.35, f'MSE = {mse:.2f} dB^2', transform=plt.gca().transAxes)
        plt.text(0.03, 0.05, f'RMSE = {rmse:.2f} dB', transform=plt.gca().transAxes)
        # Source for displaying text inside the plot using transAxes:
        # https://stackoverflow.com/questions/42932908/matplotlib-coordinates-tranformation
        
    # Adjust layout to prevent overlap
    plt.tight_layout()

# Keep all figures open
plt.show(block=True)