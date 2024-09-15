# Panos Lelakis, 1083712, Headphone EQ, task 7

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import sosfilt, sosfreqz, tf2sos, find_peaks
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

# Function to detect bells in the desired filter response plot
def detect_bells(signal, freqs, threshold=0.1, min_distance=10, min_prominence=0.01):
    # Detect bells in the range 20 Hz - 20 kHz
    low_index = np.where(freqs >= 20)[0][0]
    high_index = np.where(freqs <= 20000)[0][-1]
    signal = signal[low_index:high_index + 1]
    freqs = freqs[low_index:high_index + 1]
    
    # Sub-function to detect bells
    def find_bells(peaks, properties, signal, freqs):
        bells = []

        for i, peak in enumerate(peaks):
            peak_gain = properties['peak_heights'][i]

            # To calculate the desired Q factor, the left and right frequencies of the bell are needed.
            # Based on the formula from the source:
            # https://ecstudiosystems.com/discover/textbooks/basic-electronics/filters/quality-factor-and-bandwidth/

            # Start and end of the 3dB bandwidth
            W1 = None
            W2 = None

            # Find W1 (left side of the peak)
            for j in range(peak, 0, -1):
                if signal[j] <= peak_gain - 3:
                    W1 = freqs[j]
                    break

            # Find W2 (right side of the peak)
            for j in range(peak, len(signal)):
                if signal[j] <= peak_gain - 3:
                    W2 = freqs[j]
                    break

            # Calculate the Q factor if W1 and W2 exist
            if ((W1 is not None) and (W2 is not None) and (W2 > W1)):
                Qo = freqs[peak] / (W2 - W1)
            else:
                continue

            bell = {
                'Wo': freqs[peak],
                'gain': peak_gain,
                'Qo': Qo,
                'left_index': W1,
                'right_index': W2
            }

            bells.append(bell)

        return bells

    # Source for finding peak point and range of a bell:
    # https://stackoverflow.com/questions/56810147/find-closest-minima-to-peak/

    # Find positive peaks (positive bells)
    positive_peaks, positive_properties = find_peaks(signal, height=threshold, distance=min_distance, prominence=min_prominence)
    positive_bells = find_bells(positive_peaks, positive_properties, signal, freqs)

    # Find negative peaks (negative bells)
    negative_peaks, negative_properties = find_peaks(-signal, height=threshold, distance=min_distance, prominence=min_prominence)
    negative_bells = find_bells(negative_peaks, negative_properties, -signal, freqs)  # Use -signal to identify negative bells (as positive)

    # Combine positive and negative bells
    bells = positive_bells + negative_bells

    # Remove overlapping bells based on their gain
    filtered_bells = [] # filtered_bells contains the bells that will be kept
    for bell in bells:
        overlap_found = False
        
        for existing_bell in filtered_bells:
            # Check if the current bell overlaps with any existing bell
            if (bell['left_index'] <= existing_bell['right_index'] and bell['right_index'] >= existing_bell['left_index']):
                overlap_found = True
                # If the current bell overlaps with an existing one, keep the one with the bigger gain
                if abs(bell['gain']) >= abs(existing_bell['gain']):
                    filtered_bells.remove(existing_bell)
                    filtered_bells.append(bell)
                break
        
        # If the current bell doesn't overlap with any existing bell, add it to the filtered list
        if not overlap_found:
            filtered_bells.append(bell)

    # Gain was calculated as positive for the negative bells (due to -signal), so replace with negative gain
    for bell in negative_bells:
        bell['gain'] = -bell['gain']

    return filtered_bells

# Function to detect sloped curves in the desired filter response plot
def detect_slopes(signal, freqs, bells):
    slopes = []
    indices = [] # indices list holds the indices at which the shelf slope will be calculated

    # Identify the first (minf) and last (maxf) bell center frequencies
    if bells:
        minf = bells[0]['Wo']
        maxf = bells[0]['Wo']
        for bell in bells:
            if bell['Wo'] < minf:
                minf = bell['Wo']
            if bell['Wo'] > maxf:
                maxf = bell['Wo']
    
    indices.append(np.where(freqs >= 20)[0][0])
    indices.append(np.where(freqs == minf)[0][0])
    indices.append(np.where(freqs <= 20000)[0][-1])
    indices.append(np.where(freqs == maxf)[0][-1])

    # Calculate slopes for the regions: 20 Hz - minf and maxf - 20 kHz
    for i in range(2):
        slope = (signal[indices[2*i+1]] - signal[indices[2*i]]) / (freqs[indices[2*i+1]] - freqs[indices[2*i]]) # Slope of a simple line
        S = np.cos(np.arctan(slope)) # Slope is the tangent of the angle of the line, so get the angle and use its cosine as slope
        if i == 0: # For the first (starting) frequency range, boost or attenuate the left (low) frequencies -> lowshelf
            type = 'lowshelf'
        else: # For the last (ending) frequency range, boost or attenuate the right (high) frequencies -> highshelf
            type = 'highshelf'
        Wo = (freqs[indices[2*i+1]] + freqs[indices[2*i]]) / 2  # Geometric mean for center frequency
        gain =  (signal[indices[2*i+1]] + signal[indices[2*i]]) / 2 # Average gain

        slopes.append({
            'type': type,
            'Wo': Wo,
            'S': S,
            'gain': gain
        })

    return slopes

# Function to apply required filters after detecting bells and slopes
def apply_filters(hp, fs, bells, slopes, message):
    filters = [] # filters list holds all the filters applied to the signal
    counter = 0 # Filter counter
    
    for bell in bells: # For each detected bell, apply a corresponding bell filter
        Wo = bell['Wo']
        Qo = bell['Qo']
        gain = bell['gain']
        sos = design_parametric([Wo], [Qo], [gain], fs, ['bell'])
        filters.append(sos)
        counter += 1
        print(f'bell filter at {Wo:.1f} Hz with Q = {Qo:.2f} and Gain = {gain:.2f}')
    
    for slope in slopes:
        Wo = slope['Wo']
        S = slope['S']
        gain = slope['gain']
        type = slope['type']
        sos = design_parametric([Wo], [S], [gain], fs, [type])
        filters.append(sos)
        counter += 1
        print(f'{type} filter at {Wo:.1f} Hz with S = {S:.2f} and Gain = {gain:.2f}')

    # Apply all filters and return the combined SOS
    combined_sos = np.vstack(filters)
    filtered_hp = sosfilt(combined_sos, hp)
    
    print(f'{message}')
    print(f'Total number of filters: {counter}')
    print('\n-----------------------------------\n\n')

    return combined_sos, filtered_hp

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
        
        # Calculate desired filter frequency response
        desired_mag = target_mag / hp_mag
        desired_mag_db = 20 * np.log10(desired_mag)
        desired_freq = target_freq

        # Detect bells and slopes
        bells = detect_bells(desired_mag_db, desired_freq, threshold=0.1, min_distance=10, min_prominence=0.01)
        slopes = detect_slopes(desired_mag_db, desired_freq, bells)

        # Apply filters and get overall response
        combined_sos, filtered_hp = apply_filters(hp[i], hp_fs[i], bells, slopes, f'Headphone Response {i+1} vs {legend}')
        
        # Calculate frequency response of filtered headphone
        filtered_hp_fft = np.fft.rfft(filtered_hp, len(filtered_hp), norm='forward')
        filtered_hp_mag = np.abs(filtered_hp_fft)/2
        filtered_hp_mag_db = 20 * np.log10(filtered_hp_mag) - scaling_db
        filtered_hp_freq = np.linspace(0, hp_fs[i] / 2, len(filtered_hp_mag_db))
        
        # Calculate frequency response of actual filter
        w, h = sosfreqz(combined_sos, worN=len(desired_mag))
        h_mag = np.abs(h)
        h_mag_db = 20 * np.log10(h_mag + 1e-12)
        h_freq = 0.5 * hp_fs[i] * w / np.pi

        # Calculate new MSE and RMSE
        mse = np.mean((filtered_hp_mag - target_mag) ** 2)
        rmse = np.sqrt(mse)

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
        # Display new MSE and RMSE inside the plot
        plt.text(0.03, 0.35, f'MSE = {mse:.2f} dB^2', transform=plt.gca().transAxes)
        plt.text(0.03, 0.05, f'RMSE = {rmse:.2f} dB', transform=plt.gca().transAxes)

    # Adjust layout to prevent overlap    
    plt.tight_layout()

# Keep all figures open
plt.show(block=True)