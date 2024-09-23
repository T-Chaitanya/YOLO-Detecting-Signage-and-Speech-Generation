from io import BytesIO
import pygame
pygame.init()
import os
from gtts import gTTS

from pygame import mixer




def speech_out(text=""):
    tts = gTTS(text=text, lang='en', slow=False, lang_check=False)
    tts.save("from_file.mp3")
    os.system(f'afplay from_file.mp3')
    # mixer.init()
    # mixer.music.load("from_file.mp3")
    # my_sound = pygame.mixer.Sound('from_file.mp3')
    # my_sound.play()

speech_out("hi chinni")