import torch
torch.set_num_threads(1)
from pprint import pprint

# Function to calculate Speech Ratio in the input Audio

""" Here I'm using Silero Voice activity Detection (Silero-vad) library
    to detect the time stampings of speech in the audio 
    
    Using these time stampings, I'm calculating the total Speech duration.

    Speech Ratio = (Speech duration / Total duration) x 100 """

def speech_ratio_calculation_function(audio_path):

    try:

        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True)

        (get_speech_timestamps,
        _, read_audio,
        *_) = utils

        sampling_rate = 16000 # also accepts 8000
        wav = read_audio(audio_path, sampling_rate=sampling_rate)

        # Calling get_speech_timestamps() function from silero-vad libray
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
        
        pprint(speech_timestamps)

        speech_duration = 0

        for x in speech_timestamps:
            diff = x['end'] - x['start']
            speech_duration = speech_duration + diff

        speech_ratio = (speech_duration / len(wav)) * 100

        return speech_ratio
    
    except Exception as err:
        
        print("Error in computing Speech Ratio => ", err)
        return 0