import snr
import noise_reduction as nr
import speech_ratio as sr

isInValidChoice = True

while (isInValidChoice):
    print("""
          1. Speech Ratio calculation
          2. Noise reduction
          3. SNR calculation
          """)
    
    option = int(input("Please choose the task you want to test: "))

    if option == 1 or option == 2 or option == 3:
        isInValidChoice = False
    else:
        print("You have choosen a Invalid method!")

audio_path = input("Please enter the path of the audio file including the extension: \n")

if option == 1:    
    speech_ratio_value = sr.speech_ratio_calculation_function(audio_path)
    print("Speech Ratio = ", speech_ratio_value)

elif option == 2:
    nr.noise_reduction_function(audio_path)

elif option == 3:
    snr_db = snr.snr_calculation_function(audio_path)
    print("SNR = ", snr_db, " dB")