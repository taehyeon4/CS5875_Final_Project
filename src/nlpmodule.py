import time
from datetime import datetime
from dotenv import load_dotenv
import os
#import language_tool_python
import openai
from gtts import gTTS
import pygame

load_dotenv()

class NLPModule:
    def __init__(self):
        #self.language_tool = language_tool_python.LanguageTool('en-US')
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.audio_logs_dir = "results/audio_logs"
        os.makedirs(self.audio_logs_dir, exist_ok=True)
        pygame.mixer.init()

    def fix_grammar(self, input_text):
        input_text = input_text.strip().capitalize()
        corrected_text = self.language_tool.correct(input_text)
        return corrected_text

    def remove_profanity(self, input_text):
        prompt = (
            f"Rewrite the following sentence to remove all profanity or offensive language "
            f"while keeping the meaning clear and appropriate:\n\n"
            f"Input: {input_text}\nOutput:"
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.6,
        )
        
        cleaned_text = response['choices'][0]['message']['content'].strip()
        return cleaned_text

    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='en')
        curr_string_name = datetime.now().strftime("%b%d%Y%H%M%S")
        file_path = os.path.join(self.audio_logs_dir, f"{curr_string_name}.mp3")
        tts.save(file_path)
        return file_path

    def play_audio(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    def process_text(self, input_text):
        #grammar_fixed = self.fix_grammar(input_text)
        no_profanity = self.remove_profanity(input_text)
        audio_file_path = self.text_to_speech(no_profanity)
        self.play_audio(audio_file_path)
        return no_profanity