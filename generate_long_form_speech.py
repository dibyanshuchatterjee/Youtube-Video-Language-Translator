"""This script tackles the problem of long texts' conversion to text. It breaks down the text into sentences using
nltk and then generates the text semantic tokens to finally create the audio array """
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
import scipy

nltk.download('punkt')
preload_models()


def generate_speech(script):
    sentences = nltk.sent_tokenize(script)
    GEN_TEMP = 0.6
    SPEAKER = "v2/hi_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        print(sentence)
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # this controls how likely the generation is to end
        )

        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER, )
        pieces += [audio_array, silence.copy()]
    scipy.io.wavfile.write("translated_voice.wav", rate=SAMPLE_RATE, data=np.concatenate(pieces))

