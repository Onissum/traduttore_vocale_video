import os
import srt
import subprocess
import numpy as np
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import whisper
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
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
        # LanguageTool for both languages
        self.en_tool = language_tool_python.LanguageTool('en-US')
        self.it_tool = language_tool_python.LanguageTool('it')
        
        # Using lighter open source models
        self.en_model_name = "felixfd/style-t5-small"
        self.it_model_name = "gsarti/it5-small"
        
        # Cache initialization
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
        """
        Generate unique key for caching
        """
        return hashlib.md5(f"{text}:{language}".encode()).hexdigest()
    
    def get_from_cache(self, text, language):
        """
        Look for a correction in cache
        """
        cache_key = self.get_cache_key(text, language)
        self.cursor.execute(
            "SELECT corrected_text, corrections FROM corrections WHERE text_hash = ?", 
            (cache_key,)
        )
        result = self.cursor.fetchone()
        if result:
            return {
                "corrected": result[0],
                "corrections": json.loads(result[1])
            }
        return None
    
    def save_to_cache(self, text, corrected_text, corrections, language):
        """
        Save a correction to cache
        """
        cache_key = self.get_cache_key(text, language)
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO corrections 
            (text_hash, original_text, corrected_text, language, corrections, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cache_key, text, corrected_text, language, 
             json.dumps(corrections), datetime.now())
        )
        self.conn.commit()
    
    def check_text(self, text, language='en'):
        """
        Perform text correction with caching
        """
        # Check cache first
        cached = self.get_from_cache(text, language)
        if cached:
            return cached
        
        # Select tool based on language
        tool = self.en_tool if language == 'en' else self.it_tool
        
        # Correction with LanguageTool
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
        
        # Save to cache
        self.save_to_cache(text, corrected, corrections, language)
        
        return {
            "corrected": corrected,
            "corrections": corrections
        }

def process_segments(segments, checker):
    """
    Process segments with reporting
    """
    total_corrections = {"en": 0, "it": 0}
    correction_types = {"en": {}, "it": {}}
    
    for segment in segments:
        # English correction
        en_result = checker.check_text(segment["text"], "en")
        segment["original_text"] = segment["text"]
        segment["text"] = en_result["corrected"]
        segment["en_corrections"] = en_result["corrections"]
        
        # Update statistics
        total_corrections["en"] += len(en_result["corrections"])
        for corr in en_result["corrections"]:
            correction_types["en"][corr["error"]] = correction_types["en"].get(corr["error"], 0) + 1
        
        # If present, Italian correction
        if "translation" in segment:
            it_result = checker.check_text(segment["translation"], "it")
            segment["original_translation"] = segment["translation"]
            segment["translation"] = it_result["corrected"]
            segment["it_corrections"] = it_result["corrections"]
            
            total_corrections["it"] += len(it_result["corrections"])
            for corr in it_result["corrections"]:
                correction_types["it"][corr["error"]] = correction_types["it"].get(corr["error"], 0) + 1
    
    # Generate report
    generate_correction_report(segments, total_corrections, correction_types)
    
    return segments

def generate_correction_report(segments, totals, types):
    """
    Generate detailed correction report
    """
    with open('correction_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Correction Report ===\n\n")
        
        # General statistics
        f.write(f"Total corrections:\n")
        f.write(f"- English: {totals['en']}\n")
        f.write(f"- Italian: {totals['it']}\n\n")
        
        # Most common correction types
        f.write("Most frequent correction types:\n")
        for lang in ['en', 'it']:
            f.write(f"\n{lang.upper()}:\n")
            sorted_types = sorted(types[lang].items(), key=lambda x: x[1], reverse=True)
            for rule, count in sorted_types[:10]:
                f.write(f"- {rule}: {count}\n")
        
        # Segment details
        f.write("\nCorrection details by segment:\n")
        for i, segment in enumerate(segments, 1):
            f.write(f"\nSegment {i}:\n")
            f.write(f"Original text: {segment['original_text']}\n")
            f.write(f"Corrected text: {segment['text']}\n")
            if "translation" in segment:
                f.write(f"Corrected translation: {segment['translation']}\n")

def transcribe_audio_with_whisper(audio_path, model_size="base"):
    """
    Transcribe audio using Whisper
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["segments"]

def translate_segments(segments, translator):
    """
    Translate segments using MarianMT
    """
    model_name = "Helsinki-NLP/opus-mt-en-it"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    for segment in segments:
        inputs = tokenizer(segment["text"], return_tensors="pt", padding=True)
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
        # Save segment to temporary file
        temp_segment = f"temp_segment_{hash(segment['translation'])}.wav"
        engine.save_to_file(segment["translation"], temp_segment)
        engine.runAndWait()
        
        # Load and add to combined audio
        segment_audio = AudioSegment.from_wav(temp_segment)
        combined += segment_audio
        
        # Update timing
        segment["start"] = current_position
        current_position += len(segment_audio)
        segment["end"] = current_position
        
        os.remove(temp_segment)
    
    combined.export(output_path, format="mp3")
    return segments

def generate_subtitles(segments, output_path):
    """
    Generate SRT subtitles with translated text only.
    """
    subs = []
    for i, segment in enumerate(segments, 1):
        start = timedelta(milliseconds=segment["start"])
        end = timedelta(milliseconds=segment["end"])
        
        # Include only the translated text
        subs.append(srt.Subtitle(
            index=i,
            start=start,
            end=end,
            content=segment["translation"]  # Only the translated text
        ))
    
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
        segment.write_videofile(segment_path)
        segments.append(segment_path)
    
    video.close()
    return segments

def add_audio_and_subtitles_to_video(input_video, audio_path, subtitle_path, output_path):
    """
    Combine video with new audio and subtitles
    """
    cmd = [
        'ffmpeg', '-i', input_video,
        '-i', audio_path,
        '-vf', f"subtitles={subtitle_path}:force_style='FontName=Arial,Fontsize=18,PrimaryColour=&H00FF00&,BackColour=&H000000&'",
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-c:a', 'aac',
        output_path
    ]
    subprocess.run(cmd, check=True)

def merge_videos(segment_paths, output_path):
    """
    Merge processed video segments
    """
    clips = [VideoFileClip(path) for path in segment_paths]
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(output_path)
    
    for clip in clips:
        clip.close()
    
    for path in segment_paths:
        os.remove(path)

def process_single_segment(input_video_path, output_video_path, model_size="base", target_language="it", speed=1.0):
    """
    Process a single video segment
    """
    temp_audio_path = f"temp_audio_{os.path.basename(input_video_path)}.wav"
    temp_translated_audio = f"temp_translated_{os.path.basename(input_video_path)}.mp3"
    temp_subtitle_path = f"temp_subs_{os.path.basename(input_video_path)}.srt"

    try:
        print("Extracting audio from segment...")
        video = VideoFileClip(input_video_path)
        video.audio.write_audiofile(temp_audio_path, fps=16000, ffmpeg_params=["-ac", "1"])
        video.close()

        segments = transcribe_audio_with_whisper(temp_audio_path, model_size)
        segments = translate_segments(segments, None)  # Translator instance not needed

        # Grammar correction
        checker = OpenSourceGrammarChecker()
        segments = process_segments(segments, checker)

        segments = generate_audio_with_timing(segments, temp_translated_audio, speed)
        generate_subtitles(segments, temp_subtitle_path)

        add_audio_and_subtitles_to_video(
            input_video_path,
            temp_translated_audio,
            temp_subtitle_path,
            output_video_path
        )

    finally:
        # Cleanup
        for temp_file in [temp_audio_path, temp_translated_audio, temp_subtitle_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def process_video_segments(input_video_path, output_video_path, model_size="base", target_language="it", speed=1.0):
    """
    Process complete video by segments
    """
    try:
        video_segments = split_video(input_video_path)
        processed_segments = []
        
        for i, segment_path in enumerate(video_segments):
            print(f"\nProcessing segment {i+1}/{len(video_segments)}")
            
            segment_output = f"temp_processed_segment_{i}.mp4"
            process_single_segment(
                segment_path,
                segment_output,
                model_size,
                target_language,
                speed
            )
            processed_segments.append(segment_output)
            os.remove(segment_path)
        
        merge_videos(processed_segments, output_video_path)
        
    except Exception as error:
        print(f"Error during segment processing: {error}")
        for path in video_segments + processed_segments:
            if os.path.exists(path):
                os.remove(path)
        raise

if __name__ == "__main__":
    input_video = "input_video.mp4"
    output_video = "output_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: File {input_video} does not exist.")
    else:
        try:
            process_video_segments(
                input_video_path=input_video,
                output_video_path=output_video,
                model_size="base",
                target_language="it",
                speed=1.0
            )
            print("Processing completed successfully!")
        except Exception as error:
            print(f"An error occurred during processing: {error}")
