import os
import srt
import subprocess
import numpy as np
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from transformers import MarianMTModel, MarianTokenizer
from pydub import AudioSegment
import pyttsx3
import math
import language_tool_python
import sqlite3
import hashlib
import json

class OpenSourceGrammarChecker:
    def __init__(self):
        """
        Initialize the correction system using open source tools
        """
        self.en_tool = language_tool_python.LanguageTool('en-US')
        self.it_tool = language_tool_python.LanguageTool('it')
        self.setup_cache()
        
    def setup_cache(self):
        """
        Create SQLite database for corrections caching
        """
        self.conn = sqlite3.connect('corrections_cache.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                text_hash TEXT PRIMARY KEY,
                original_text TEXT,
                corrected_text TEXT,
                language TEXT,
                corrections TEXT,
                timestamp DATETIME
            )
        ''')
        self.conn.commit()
    
    def get_cache_key(self, text, language):
        return hashlib.md5(f"{text}:{language}".encode()).hexdigest()
    
    def get_from_cache(self, text, language):
        cache_key = self.get_cache_key(text, language)
        self.cursor.execute(
            "SELECT corrected_text, corrections FROM corrections WHERE text_hash = ?", 
            (cache_key,)
        )
        result = self.cursor.fetchone()
        if result:
            return {"corrected": result[0], "corrections": json.loads(result[1])}
        return None
    
    def save_to_cache(self, text, corrected_text, corrections, language):
        cache_key = self.get_cache_key(text, language)
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO corrections 
            (text_hash, original_text, corrected_text, language, corrections, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cache_key, text, corrected_text, language, json.dumps(corrections), datetime.now())
        )
        self.conn.commit()
    
    def check_text(self, text, language='en'):
        cached = self.get_from_cache(text, language)
        if cached:
            return cached
        
        tool = self.en_tool if language == 'en' else self.it_tool
        matches = tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        
        corrections = [
            {
                "error": match.ruleId,
                "message": match.message,
                "context": match.context,
                "suggestions": match.replacements[:3]
            }
            for match in matches
        ]
        
        self.save_to_cache(text, corrected, corrections, language)
        return {"corrected": corrected, "corrections": corrections}
    
def transcribe_audio_with_whisper(audio_path, model_size="base"):
    """
    Transcribe audio using Whisper
    """
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    print(f"Transcription complete. Total segments: {len(result['segments'])}")
    return result["segments"]

def translate_segments(segments, model_name="Helsinki-NLP/opus-mt-en-it"):
    """
    Translate segments using MarianMT
    """
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    for segment in segments:
        text = segment.get("text", "")  # Recupera il testo originale
        if not text.strip():  # Ignora segmenti vuoti
            segment["translation"] = ""
            continue

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        segment["translation"] = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return segments

def generate_audio_with_timing(segments, output_path, speed=1.0):
    """
    Generate audio for translated segments
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', int(engine.getProperty('rate') * speed))
    
    combined = AudioSegment.empty()
    current_position = 0
    
    for segment in segments:
        temp_segment = f"temp_segment_{hash(segment['translation'])}.wav"
        engine.save_to_file(segment["translation"], temp_segment)
        engine.runAndWait()
        
        segment_audio = AudioSegment.from_wav(temp_segment)
        combined += segment_audio
        
        segment["start"] = current_position
        current_position += len(segment_audio)
        segment["end"] = current_position
        
        os.remove(temp_segment)
    
    combined.export(output_path, format="mp3")
    return segments

def generate_subtitles(segments, output_path):
    """
    Generate SRT subtitles with translated text
    """
    subs = []
    for i, segment in enumerate(segments, 1):
        start = timedelta(milliseconds=segment["start"])
        end = timedelta(milliseconds=segment["end"])
        subs.append(srt.Subtitle(index=i, start=start, end=end, content=segment["translation"]))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))
def split_video(input_path, segment_duration=300):
    """
    Split video into manageable segments
    """
    video = VideoFileClip(input_path)
    duration = video.duration
    segments = []
    
    for start in range(0, math.ceil(duration), segment_duration):
        end = min(start + segment_duration, duration)
        segment = video.subclip(start, end)
        segment_path = f"temp_segment_{start}.mp4"
        segment.write_videofile(segment_path, codec="libx264")
        segments.append(segment_path)
    
    video.close()
    return segments

def add_audio_and_subtitles_to_video(input_video, audio_path, subtitle_path, output_path):
    """
    Combine video with new audio and subtitles
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_video,        # Input video
            '-i', audio_path,                  # Input audio
            '-vf', f"subtitles={subtitle_path}:force_style='FontName=Arial,Fontsize=18,PrimaryColour=&H00FF00&,BackColour=&H000000&'",
            '-map', '0:v', '-map', '1:a',      # Map video from input 0 and audio from input 1
            '-c:v', 'libx264',                 # Codec for video
            '-c:a', 'aac',                     # Codec for audio
            '-strict', 'experimental',         # Enable experimental features
            output_path                        # Output file
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while combining video, audio, and subtitles: {e}")
        raise

def process_single_video(input_video, output_video, model_size="base", target_language="it", speed=1.0):
    """
    Process a single video, translating and adding subtitles and audio
    """
    temp_audio = "temp_audio.wav"
    temp_translated_audio = "temp_translated.mp3"
    temp_subtitles = "temp_subtitles.srt"
    
    try:
        print(f"Processing: {input_video}")
        video = VideoFileClip(input_video)
        video.audio.write_audiofile(temp_audio, fps=16000, ffmpeg_params=["-ac", "1"])
        video.close()
        
        print("Transcribing audio...")
        segments = transcribe_audio_with_whisper(temp_audio, model_size)
        
        print("Translating segments...")
        segments = translate_segments(segments)  # Aggiunge la chiave `translation`
        
        print("Generating audio and subtitles...")
        generate_audio_with_timing(segments, temp_translated_audio, speed)
        generate_subtitles(segments, temp_subtitles)
        
        print("Combining video, audio, and subtitles...")
        add_audio_and_subtitles_to_video(input_video, temp_translated_audio, temp_subtitles, output_video)
    
    finally:
        for temp_file in [temp_audio, temp_translated_audio, temp_subtitles]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def process_all_videos(input_dir, output_dir, model_size="base", target_language="it", speed=1.0):
    """
    Process all videos in the input directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    for video in video_files:
        input_video = os.path.join(input_dir, video)
        output_video = os.path.join(output_dir, f"processed_{video}")
        print(f"Processing video: {input_video}")
        process_single_video(input_video, output_video, model_size, target_language, speed)
        print(f"Video processed: {output_video}")

if __name__ == "__main__":
    input_directory = "video_da_tradurre"
    output_directory = "video_tradotti"
    process_all_videos(input_directory, output_directory, model_size="base", target_language="it", speed=1.0)
