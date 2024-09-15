# Panos Lelakis, 1083712, Headphone EQ, task 6

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import sosfilt, sosfreqz, tf2sos
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

# Function to desgin SOS filter, based on the MATLAB designParamEQ function. Source:
# https://www.mathworks.com/help/audio/ug/parametric-equalizer-design.html
def design_parametric(Wo, Qo, gain, fs, filter_type):
    if not (len(Wo) == len(Qo) and len(Wo) == len(gain) and len(Wo) == len(filter_type)):
        raise ValueError('Error: all input arrays must have same length')
    
    filters = []

    for i in range(len(Wo)):
        wo = 2 * np.pi * Wo[i] / fs
        alpha = np.sin(wo) / (2 * Qo[i])
        A = 10 ** (gain[i] / 40)

        # Get coefficients depending on the filter type
        if filter_type[i] == 'notch':
            b0 = 1
            b1 = -2 * np.cos(wo)
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * np.cos(wo)
            a2 = 1 - alpha
        
        elif filter_type[i] == 'lowpass':
            b0 = (1 - np.cos(wo)) / 2
            b1 = 1 - np.cos(wo)
            b2 = (1 - np.cos(wo)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(wo)
            a2 = 1 - alpha
        
        elif filter_type[i] == 'highpass':
            b0 = (1 + np.cos(wo)) / 2
            b1 = -(1 + np.cos(wo))
            b2 = (1 + np.cos(wo)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(wo)
            a2 = 1 - alpha
        
        elif filter_type[i] == 'bandpass':
            b0 = np.sin(wo) / 2
            b1 = 0
            b2 = -np.sin(wo) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(wo)
            a2 = 1 - alpha
        
        elif filter_type[i] == 'allpass':
            b0 = 1 - alpha
            b1 = -2 * np.cos(wo)
            b2 = 1 + alpha
            a0 = 1 + alpha
            a1 = -2 * np.cos(wo)
            a2 = 1 - alpha
        
        elif filter_type[i] == 'lowshelf':
            # For shelving filters, alpha is different. In this case, the input Qo is S (Slope)
            alpha = np.sin(wo) / 2 * np.sqrt((A + 1 / A) * (1 / Qo[i] - 1) + 2)
            b0 = A * ((A + 1) - (A - 1) * np.cos(wo) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(wo))
            b2 = A * ((A + 1) - (A - 1) * np.cos(wo) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(wo) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(wo))
            a2 = (A + 1) + (A - 1) * np.cos(wo) - 2 * np.sqrt(A) * alpha
        
        elif filter_type[i] == 'highshelf':
            alpha = np.sin(wo) / 2 * np.sqrt((A + 1 / A) * (1 / Qo[i] - 1) + 2)
            b0 = A * ((A + 1) + (A - 1) * np.cos(wo) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(wo))
            b2 = A * ((A + 1) + (A - 1) * np.cos(wo) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(wo) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(wo))
            a2 = (A + 1) - (A - 1) * np.cos(wo) - 2 * np.sqrt(A) * alpha
        
        elif filter_type[i] == 'bell': # Bell (peaking) filter
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(wo)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(wo)
            a2 = 1 - alpha / A
        
        else:
            raise ValueError('Unsupported filter type')
        
        b = [b0, b1, b2]
        a = [a0, a1, a2]
        # Normalize coefficients:
        b = np.array(b) / a0
        a = np.array(a) / a0

        # Formulas taken from AudioeqCookbook - source:
        # https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
        
        filters.append(tf2sos(b, a)) # tf2sos transforms b,a coefficients to SOS
    
    # Source for using vstack to stack vertically filters:
    # https://scipython.com/book/chapter-6-numpy/examples/vstack-and-hstack/
    return np.vstack(filters)

# Plot magnitude responses
for i in range(hpfilesnum):
    # Calculate frequency response for headphone
    hp_fft = np.fft.rfft(hp[i], len(hp[i]), norm='forward')  # norm='forward' for 1/n scaling
    hp_mag = np.abs(hp_fft)/2 # Divide by 2 for further scaling
    hp_freq = np.linspace(0, hp_fs[i]/2, len(hp_mag))
    hp_mag_db = 20 * np.log10(hp_mag) - scaling_db

    plt.figure(i + 1)
    
    # Plot target responses in frequency domain
    for j in range(targetsnum):
        # Calculate frequency response of target
        target_fft = np.fft.rfft(target[j], len(hp[i]), norm='forward')
        target_mag = np.abs(target_fft)/2
        target_freq = np.linspace(0, target_fs[j]/2, len(target_mag))
        target_mag_db = 20 * np.log10(target_mag) - scaling_db
        
        # Calculate initial MSE and RMSE
        mse = np.mean((hp_mag - target_mag) ** 2)
        rmse = np.sqrt(mse)
        # Source for mse and rmse formulas:
        # https://en.wikipedia.org/w/index.php?title=Mean_squared_error&oldid=1207422018

        # Plot headphone response vs target response
        plt.subplot(3, targetsnum, j + 1)
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
        plt.text(0.03, 0.35, f'MSE = {mse:.2f} dB^2', transform=plt.gca().transAxes)
        plt.text(0.03, 0.05, f'RMSE = {rmse:.2f} dB', transform=plt.gca().transAxes)
        # Source for changing color and linewidth for specific gridlines:
        # https://stackoverflow.com/questions/32073616/matplotlib-change-color-of-individual-grid-lines
        # Source for making all ticks visible using ScalarFormatter():
        # https://www.reddit.com/r/learnpython/comments/epwteg/matplotlib_ticklabels_disappearing_in_log_scale/
        # Source for displaying text inside the plot using transAxes:
        # https://stackoverflow.com/questions/42932908/matplotlib-coordinates-tranformation

        # Example usage of design_parametric function
        tmpsos = design_parametric([18000, 100, 1000, 10000], [1, 0.8, 1, 2], [1, 30, 10, -10], hp_fs[i], ['lowpass', 'bell', 'iirpeak', 'bell'])
        
        # Apply the parametric equalizer to the headphone response
        filtered_hp = sosfilt(tmpsos, hp[i])
        
        # Calculate frequency response of filtered headphone
        filtered_hp_fft = np.fft.rfft(filtered_hp, len(filtered_hp), norm='forward')
        filtered_hp_mag = np.abs(filtered_hp_fft)/2
        filtered_hp_mag_db = 20 * np.log10(filtered_hp_mag) - scaling_db
        filtered_hp_freq = np.linspace(0, hp_fs[i] / 2, len(filtered_hp_mag_db))

        # Calculate desired filter frequency response
        desired_mag = target_mag / hp_mag
        desired_mag_db = 20 * np.log10(desired_mag)
        desired_freq = target_freq
        
        # Calculate frequency response of actual filter
        w, h = sosfreqz(tmpsos, worN=len(desired_mag))
        h_mag = np.abs(h)
        h_mag_db = 20 * np.log10(h_mag + 1e-12)
        h_freq = 0.5 * hp_fs[i] * w / np.pi

        # Plot desired filter response vs actual filter response
        plt.subplot(3, targetsnum, j + 3)
        plt.plot(desired_freq, desired_mag_db, color='g')
        plt.plot(h_freq, h_mag_db, color='tab:orange')
        plt.gca().set_xscale('log')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.gca().set_yticks([-80, -50, -20, 0, 20, 40])
        plt.xlim([20, 20000])
        plt.ylim([-80, 40])
        plt.title('Desired Filter Response vs Actual Filter Response')
        plt.legend(['Desired Filter Response', 'Actual Filter Response'], loc='lower right')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.gca().grid(which='major', linestyle='--', linewidth=0.7, color='tab:brown')
        plt.gca().grid(which='minor', linestyle=':', linewidth=0.5, color='tab:gray')
        plt.gca().get_ygridlines()[3].set_color('k')
        plt.gca().get_ygridlines()[3].set_linewidth(1)
        plt.gca().minorticks_on()
        
        # Plot filtered headphone response vs target response
        plt.subplot(3, targetsnum, 5 + j)
        plt.plot(filtered_hp_freq, filtered_hp_mag_db, color='b')
        plt.plot(target_freq, target_mag_db, color='r')
        plt.gca().set_xscale('log')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.gca().set_yticks([-80, -50, -20, 0, 20, 40])
        plt.xlim([20, 20000])
        plt.ylim([-80, 40])
        legend = 'Flat Target' if j == 0 else 'Harman Target'
        plt.title(f'Filtered Headphone Response vs {legend} Response')
        plt.legend(['Filtered Headphone Response', legend], loc='lower right')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.gca().grid(which='major', linestyle='--', linewidth=0.7, color='tab:brown')
        plt.gca().grid(which='minor', linestyle=':', linewidth=0.5, color='tab:gray')
        plt.gca().get_ygridlines()[3].set_color('k')
        plt.gca().get_ygridlines()[3].set_linewidth(1)
        plt.gca().minorticks_on()
        mse = np.mean((filtered_hp_mag - target_mag) ** 2)
        rmse = np.sqrt(mse)
        plt.text(0.03, 0.35, f'MSE = {mse:.2f} dB^2', transform=plt.gca().transAxes)
        plt.text(0.03, 0.05, f'RMSE = {rmse:.2f} dB', transform=plt.gca().transAxes)

    # Adjust layout to prevent overlap    
    plt.tight_layout()

# Keep all figures open
plt.show(block=True)