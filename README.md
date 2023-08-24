# YouTube Video Transcription and Translation using Tree of Thoughts Prompting

This Python script demonstrates a logical process for transcribing and translating audio content from a YouTube video. It follows a series of steps to achieve the goal of creating a translated version of the video with synchronized audio. The script utilizes several libraries and tools to achieve this.
There are two .py files, one which does video translation without suno/bark and one with does with suno/bark

# For Bark:

The script `generate_long_form_speech.py` helps to do the actual generation.
Do not do `pip install bark`, refer https://github.com/suno-ai/bark for precise details if needed.
You may also use the `requirements.txt` file provided here

## Logical Breakdown: Speech-to-Text and Text-to-Speech Translation

### Step 1: Speech-to-Text Conversion

1. **Video Download**:
   The Pytube library is used to download the highest resolution version of a YouTube video, ensuring optimal quality for both video and audio.

2. **Audio Transcription**:
   The Whisper ASR model is employed to transcribe the spoken content of the video into text. This results in an accurate representation of the speech in text format.

### Step 2: Translated Text-to-Speech Synthesis

1. **Translation using Tree of Thoughts Prompting**:
   A logical approach, known as the "Tree of Thoughts," is used to translate the transcribed text into the desired target language. This technique involves collaborative steps to ensure accurate translation and tone preservation.

2. **Text-to-Speech Synthesis**:
   The translated text is converted into synthesized speech using the gTTS (Google Text-to-Speech) library. This bridging step transforms the translated text into a human-like voice, ensuring the auditory experience matches the original speech.

3. **Audio Duration Matching**:
   To maintain synchronization between the translated audio and the video, the script ensures the duration of the translated audio matches that of the video. This prevents any audio lag or overlap.

4. **Audio Replacement in Video**:
   MoviePy is employed to replace the original audio of the video with the translated audio. This step ensures the content of the video remains unchanged while audio is presented in the target language.

## How to Use the Script:

1. Install the required libraries listed at the beginning of the script using `pip install`.
2. Replace `openai_api_key` with your OpenAI API key in the `credentials.py` file.
3. Set the `url` variable to the YouTube video URL you want to translate.
4. Run the script using `python script_name.py`.

## Conclusion:

This script showcases a comprehensive approach to transcribing and translating audio content from YouTube videos. By combining ASR, machine translation, text-to-speech synthesis, and video editing techniques, the script provides a practical solution for creating multilingual video content.
