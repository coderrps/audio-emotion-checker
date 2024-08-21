from flask import Flask, request, render_template
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO

app = Flask(__name__)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load pre-trained models from transformers
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return render_template('index.html', error="No audio file provided")

    audio_file = request.files['audio']

    try:
        # Read audio file in memory
        audio_bytes = audio_file.read()
        print("Audio file read successfully")  # Debug statement
        
        # Convert audio to WAV format using pydub
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        audio = audio.set_channels(1)  # Mono channel
        audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
        
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        print("Audio converted to WAV format")  # Debug statement
        
        # Use SpeechRecognition to transcribe audio
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Transcription: {text}")  # Debug statement
        
        # Emotion detection
        emotions = emotion_classifier(text)
        print(f"Emotions detected: {emotions}")  # Debug statement
        # Extract top emotion
        top_emotion = max(emotions[0], key=lambda x: x['score'])
        print(f"Top Emotion: {top_emotion}")  # Debug statement
        
        # Summarization
        summary = summarizer(text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
        print(f"Summary: {summary}")  # Debug statement
        
        # Render results on the same page
        return render_template('index.html', transcription=text, emotion=top_emotion['label'], summary=summary)
    
    except sr.UnknownValueError:
        return render_template('index.html', error="Could not understand audio")
    except sr.RequestError as e:
        return render_template('index.html', error=f"Speech Recognition service error: {e}")
    except Exception as e:
        # General exception to catch other potential errors
        print(f"Unexpected error: {e}")  # Debug statement
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
