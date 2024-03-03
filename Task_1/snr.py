import librosa
import numpy as np

# Function to calculte Signal to Noise Ratio (SNR)

def snr_calculation_function(audio_path):
    
    try:
        # Loading the audio data and Sampling Rate(sr) using Librosa
        audio_data, sr = librosa.load(audio_path)

        # Performing Short time Fourier transform to the audio data
        audio_data_ft = librosa.stft(audio_data)

        # Calculate Magnitude(mag) and Phase(phase)
        mag, phase = librosa.magphase(audio_data_ft)

        # Applying Filter
        filtered_audio = librosa.decompose.nn_filter(mag, aggregate= np.median, metric='cosine', width= int(librosa.time_to_frames(1, sr=sr)))
        filtered_audio = np.minimum(mag, filtered_audio)

        # Using margin values to reduce blend between speech and noise
        margin_b, margin_f = 2, 10
        power = 2

        mask_b = librosa.util.softmask(filtered_audio,
                                    margin_b * (mag - filtered_audio),
                                    power=power)

        mask_f = librosa.util.softmask(mag - filtered_audio,
                                    margin_f * filtered_audio,
                                    power=power)

        audio_foreground = mask_f * mag
        audio_background = mask_b * mag

        # Calculate signal power
        signal_power = np.mean(audio_foreground ** 2)
        noise_power = np.mean(audio_background ** 2)

        # Handle division by zero
        if noise_power == 0:
          return np.inf
        
        # Calculate SNR and convert to dB
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)

        return snr_db

    except Exception as err:
        print("Error in computing snr value => ", err)
        return 0