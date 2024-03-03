import librosa
import numpy as np
import soundfile as sf

# Function to perform Noise Reduction

def noise_reduction_function(audio_path):
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

        # Perform Short time Inverse Fourier transform
        foreground_audio = librosa.istft(audio_foreground * phase)

        sf.write('cleaned_audio.wav', foreground_audio, sr, subtype="PCM_24")
        print("Noise Reduction done !")
        print('You can find the cleaned signal saved as "cleaned_audio.wav"')

    except Exception as err:
        print("Error in computing noise reduction => ", err)