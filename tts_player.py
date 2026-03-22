import pyttsx3
import tempfile
import os

class TTSPlayer:
    def __init__(self, rate=150, volume=1.0, voice_id=None):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            if voice_id is not None:
                self.engine.setProperty('voice', voice_id)
            else:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if 'english' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.enabled = True
            print("✅ TTS 语音引擎已加载")
        except Exception as e:
            print(f"⚠️ TTS 语音加载失败: {e}")
            self.enabled = False
    
    def speak(self, text):
        if not self.enabled or not text:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS speak error: {e}")
    
    def save_to_file(self, text, filename):
        if not self.enabled or not text:
            return
        try:
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS save error: {e}")
    
    def save_to_temp(self, text, suffix=".wav"):
        if not self.enabled:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
        self.save_to_file(text, temp_path)
        return temp_path
    
    def list_voices(self):
        if not self.enabled:
            return []
        voices = self.engine.getProperty('voices')
        return [(voice.id, voice.name) for voice in voices]
    
    def set_rate(self, rate):
        if self.enabled:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        if self.enabled:
            self.engine.setProperty('volume', volume)


if __name__ == "__main__":
    tts = TTSPlayer()
    if tts.enabled:
        print("Available voices:")
        for voice_id, voice_name in tts.list_voices():
            print(f"  {voice_name}")
        tts.speak("Hello, this is a test of the text to speech system.")
    else:
        print("TTS is not available")
