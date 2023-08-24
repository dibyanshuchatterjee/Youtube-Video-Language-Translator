import googletrans
from pydub import AudioSegment
from pytube import YouTube
from moviepy.editor import *
import whisper
from langdetect import detect
import os
from googletrans import Translator
from gtts import gTTS
from credentials import openai_api_key
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI


def prepare_output_speech(text_prompt):
    language = "hi"
    # Convert translated text into speech audio
    speak = gTTS(text=text_prompt, lang=language, slow=False)
    speak.save("translated_voice.mp3")
    return True


os.environ["OPENAI_API_KEY"] = f"{openai_api_key}"
translator = Translator()

# Enter a youtube video link
url = "https://www.youtube.com/shorts/cLmD50_Li50"
yt = YouTube(url)
# Download the highest quality video
output_loc = yt.streams.get_highest_resolution().download()

# Transcribe the video
model = whisper.load_model("base")
text = model.transcribe(output_loc)['text']
print(text)

# Using Tree of thoughts prompt engineering method to translate the speech into desired language
translation_llm = OpenAI(temperature=0.7, max_tokens=3000)

# Detect source language
src_language = detect(text=text)
print(googletrans.LANGUAGES)

# Enter destination language and Translate text to destination language
dest_language = "Hindi"

# Find out the length of the video

clip = VideoFileClip(output_loc)
required_time = str(clip.duration)

pre_template = f"""Imagine three different experts are answering this question.
They will brainstorm the answer step by step reasoning carefully and taking all facts into consideration
All experts will write down 1 step of their thinking,
then share it with the group.
They will each critique their response, and the all the responses of others
They will check their answer based on science and the laws of physics
Then all experts will go on to the next step and write down this step of their thinking.
They will keep going through steps until they reach their conclusion taking into account the thoughts of the other experts
If at any time they realise that there is a flaw in their logic they will backtrack to where that flaw occurred 
If any expert realises they're wrong at any point then they acknowledges this and start another train of thought
Each expert will assign a likelihood of their current assertion being correct
Continue until the experts agree on the single most likely location
The question is...
* translate the text given in {src_language} to {dest_language} with the same tone as the original text had.
* The translated text should as such that if it is spoken in the {dest_language}, it can be completely spoken in {required_time}
* Give the final exact translated text as per the original text, that all experts agree upon. 
"""

template = pre_template + " The given text is: {text}."

# Proceed with translating into {src_language}
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=translation_llm)

translated_text = llm_chain.run(text).strip()
print(translated_text)
print(type(translated_text))


res = prepare_output_speech(text_prompt=str(translated_text))

if not res:
    print("The speech was not generated")

translated_audio = AudioFileClip("translated_voice.mp3")

# Match the duration of video and translated audio
if clip.duration > translated_audio.duration:
    padding_duration = clip.duration - translated_audio.duration
    padding_audio = AudioSegment.silent(duration=int(padding_duration * 1000))
    translated_audio = translated_audio.set_duration(clip.duration)
    translated_audio = translated_audio.overlay(padding_audio)

elif translated_audio.duration > clip.duration:
    translated_audio = translated_audio.subclip(0, clip.duration)

# translated_audio = translated_audio.set_duration(clip.duration)

# Replace the audio in the video clip
clip = clip.set_audio(translated_audio)

# Write the final video with translated audio
output_video_path = "no_bark_translated_speech_output.mp4"
clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# time.sleep(60)
# Clean up unnecessary files
os.remove('translated_voice.mp3')
os.remove(output_loc)
