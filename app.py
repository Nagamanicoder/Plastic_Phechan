# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import tempfile
import uuid
import speech_recognition as sr
import ollama
from gtts import gTTS
from pydub import AudioSegment
import io
import logging
from datetime import datetime, timedelta
import threading
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
AUDIO_FOLDER = 'temp_audio'
DATA_FILE = 'data.json'
OLLAMA_MODEL = 'gemma:2b'  # Using locally installed Gemma 2B for faster responses
MAX_AUDIO_AGE_HOURS = 24  # Hours after which audio files are cleaned up

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load plastic data
def load_plastic_data():
    """Load plastic data from JSON file"""
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Data file {DATA_FILE} not found!")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {DATA_FILE}")
        return {}

# Audio cleanup function
def cleanup_old_audio_files():
    """Remove audio files older than MAX_AUDIO_AGE_HOURS"""
    cutoff_time = datetime.now() - timedelta(hours=MAX_AUDIO_AGE_HOURS)
    
    for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                    except OSError:
                        logger.warning(f"Could not remove file: {filename}")

# Start cleanup thread
def start_cleanup_thread():
    """Start background thread for periodic cleanup"""
    def cleanup_loop():
        while True:
            time.sleep(3600)  # Run every hour
            cleanup_old_audio_files()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

# Convert audio to WAV format
def convert_to_wav(audio_file_path):
    """Convert audio file to WAV format for speech recognition"""
    try:
        # Try using pydub with ffmpeg
        audio = AudioSegment.from_file(audio_file_path)
        wav_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        logger.warning(f"FFmpeg conversion failed: {str(e)}")
        # Fallback: try to use the original file if it's already a supported format
        file_extension = audio_file_path.lower().split('.')[-1]
        if file_extension in ['wav', 'flac']:
            logger.info("Using original audio file (already in supported format)")
            return audio_file_path
        else:
            # Create a simple WAV conversion using basic method
            try:
                audio = AudioSegment.from_file(audio_file_path, format="webm")
                wav_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_path, format="wav")
                return wav_path
            except:
                logger.error("Could not convert audio format. Please install FFmpeg.")
                raise Exception("Audio conversion failed. FFmpeg required for WebM files.")

# Speech to Text function
def transcribe_audio(audio_file_path):
    """Convert audio to text using SpeechRecognition library"""
    try:
        # Convert to WAV if needed
        wav_path = convert_to_wav(audio_file_path)
        
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.2)
            # Record audio
            audio = r.record(source)
        
        # Try Google Speech Recognition (free tier)
        try:
            text = r.recognize_google(audio, language='en-IN')
            logger.info(f"Google STT successful: {text}")
            return text
        except sr.RequestError:
            logger.warning("Google STT service unavailable, trying offline recognition")
            # Fallback to offline recognition
            try:
                text = r.recognize_sphinx(audio)
                logger.info(f"Offline STT successful: {text}")
                return text
            except sr.RequestError:
                raise Exception("No speech recognition service available")
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")
            
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}")
        raise

# Find plastic data function
def find_plastic_data(transcribed_text, plastic_data):
    """Find matching plastic data based on transcribed text"""
    text_lower = transcribed_text.lower()
    
    # Simple keyword matching
    for plastic_key, plastic_info in plastic_data.items():
        # Check if plastic type is mentioned in the text
        keywords = plastic_key.replace('_', ' ').split()
        
        # Also check for common variations
        variations = {
            'water_bottle': ['water bottle', 'bottle', 'pet bottle', 'plastic bottle'],
            'milk_pouch': ['milk pouch', 'milk packet', 'milk bag', 'pouch'],
            'plastic_bag': ['plastic bag', 'shopping bag', 'carry bag', 'polythene'],
            'food_container': ['food container', 'lunch box', 'container', 'tupperware'],
            'yogurt_cup': ['yogurt cup', 'curd cup', 'dahi cup', 'cup']
        }
        
        # Check main keywords
        if any(keyword in text_lower for keyword in keywords):
            return plastic_key, plastic_info
        
        # Check variations
        if plastic_key in variations:
            if any(variation in text_lower for variation in variations[plastic_key]):
                return plastic_key, plastic_info
    
    return None, None

# Create Ollama prompt
def create_ollama_prompt(transcribed_text, plastic_key, plastic_info):
    """Create system and user prompts for Ollama"""
    
    system_prompt = """You are Plastic Pehchan Bot, helping people in Bengaluru recycle plastic properly.

Rules:
- Be friendly and encouraging
- Keep responses under 80 words
- Always mention the recycler name and address
- Include eco-points earned
- Use simple, clear language
- Focus on environmental benefits"""
    
    if plastic_info:
        user_prompt = f"""The user mentioned: "{transcribed_text}"
        
This appears to be a {plastic_key.replace('_', ' ')} which is made of {plastic_info['type']} plastic.

Please provide a helpful response that:
1. Confirms the plastic type identification
2. Tells them to take it to {plastic_info['recycler_name']} at {plastic_info['address']}
3. Mentions they can earn {plastic_info['eco_points']} Eco-Points
4. Adds an encouraging message about environmental impact

Keep the response under 100 words and make it sound natural and conversational."""
    else:
        user_prompt = f"""The user mentioned: "{transcribed_text}"

I couldn't identify the specific plastic type from their description. Please provide a helpful response that:
1. Acknowledges their request
2. Asks for more specific details about the plastic item
3. Suggests they can describe the item's shape, size, or what it contained
4. Encourages them to try again with more details
5. Mentions that proper plastic recycling helps the environment

Keep the response under 80 words and sound encouraging."""
    
    return system_prompt, user_prompt

# Get Ollama response
def get_ollama_response(system_prompt, user_prompt):
    """Get response from local Ollama instance"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.7,
                'top_p': 0.95,
                'max_tokens': 150,
                'repeat_penalty': 1.05,
                'num_ctx': 2048
            }
        )
        
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error getting Ollama response: {str(e)}")
        raise Exception(f"AI service unavailable: {str(e)}")

# Text to Speech function
def create_audio_response(text):
    """Convert text to speech and save as audio file"""
    try:
        # Create TTS object
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        
        # Generate unique filename
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        
        # Save audio file
        tts.save(audio_path)
        
        logger.info(f"Generated audio file: {audio_filename}")
        return audio_filename
        
    except Exception as e:
        logger.error(f"Error creating audio response: {str(e)}")
        raise

# Main endpoint
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Main endpoint to process audio and return response"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(f"upload_{uuid.uuid4().hex}_{audio_file.filename}")
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(upload_path)
        
        logger.info(f"Received audio file: {filename}")
        
        # Step 1: Speech to Text
        try:
            transcribed_text = transcribe_audio(upload_path)
            logger.info(f"Transcribed: {transcribed_text}")
        except Exception as e:
            return jsonify({
                'error': 'Could not process audio',
                'message': str(e)
            }), 400
        
        # Step 2: Load plastic data
        plastic_data = load_plastic_data()
        
        # Step 3: Find relevant data
        plastic_key, plastic_info = find_plastic_data(transcribed_text, plastic_data)
        
        # Step 4: Create Ollama prompt
        system_prompt, user_prompt = create_ollama_prompt(transcribed_text, plastic_key, plastic_info)
        
        # Step 5: Get Ollama response
        try:
            ai_response = get_ollama_response(system_prompt, user_prompt)
            logger.info(f"AI Response: {ai_response}")
        except Exception as e:
            return jsonify({
                'error': 'AI service error',
                'message': str(e)
            }), 500
        
        # Step 6: Text to Speech
        try:
            audio_filename = create_audio_response(ai_response)
            audio_url = f"/audio/{audio_filename}"
        except Exception as e:
            logger.warning(f"TTS failed: {str(e)}")
            # Return response without audio if TTS fails
            return jsonify({
                'text': ai_response,
                'audio_url': None,
                'message': 'Audio generation failed, but here is the text response'
            })
        
        # Clean up uploaded file
        try:
            os.remove(upload_path)
        except:
            pass
        
        # Step 7: Return response
        return jsonify({
            'text': ai_response,
            'audio_url': audio_url
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Serve audio files
@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    return send_from_directory(AUDIO_FOLDER, filename)

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ollama_model': OLLAMA_MODEL,
        'timestamp': datetime.now().isoformat()
    })

# Root endpoint
@app.route('/')
def index():
    """Root endpoint with basic info"""
    return jsonify({
        'name': 'Plastic Pehchan Backend',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            'process_audio': 'POST /process_audio - Main audio processing endpoint',
            'health': 'GET /health - Health check',
            'test': 'GET /test - Test endpoint',
            'audio': 'GET /audio/<filename> - Serve audio files'
        }
    })

# Test endpoint
@app.route('/test')
def test():
    """Test endpoint to verify server is running"""
    return jsonify({
        'message': 'Plastic Pehchan Backend is running!',
        'endpoints': ['/process_audio', '/health', '/audio/<filename>']
    })

if __name__ == '__main__':
    # Start cleanup thread
    start_cleanup_thread()
    
    # Clean up old files on startup
    cleanup_old_audio_files()
    
    print("="*50)
    print("🚀 STARTING PLASTIC PEHCHAN BACKEND")
    print(f"🤖 AI Model: {OLLAMA_MODEL}")
    print("="*50)
    
    # Test Ollama connection on startup
    try:
        test_response = ollama.list()
        print("✅ Ollama connection successful")
        if 'models' in test_response:
            available_models = []
            for model in test_response['models']:
                if isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
                elif isinstance(model, dict) and 'model' in model:
                    available_models.append(model['model'])
                else:
                    # Fallback: try to extract name from the model object
                    model_str = str(model)
                    if 'gemma:2b' in model_str or 'llama3' in model_str:
                        available_models.append(model_str)
            
            print(f"📋 Available models: {available_models}")
            
            # Check if our target model exists
            model_found = False
            for available_model in available_models:
                if OLLAMA_MODEL in available_model:
                    model_found = True
                    break
            
            if model_found:
                print(f"✅ Target model '{OLLAMA_MODEL}' found and ready!")
                # Quick test of the model
                try:
                    test_chat = ollama.chat(model=OLLAMA_MODEL, messages=[
                        {'role': 'user', 'content': 'Say hello in one word.'}
                    ])
                    print(f"🤖 Model test successful: '{test_chat['message']['content'].strip()}'")
                except Exception as e:
                    print(f"⚠️  Model test failed: {str(e)}")
            else:
                print(f"⚠️  Model '{OLLAMA_MODEL}' not clearly detected.")
                print("🔄 Will attempt to use it anyway...")
        else:
            print("📋 Model list format changed, but connection works")
            print("🔄 Will attempt to use gemma:2b...")
    except Exception as e:
        print(f"❌ Ollama connection failed: {str(e)}")
        print("Please make sure Ollama is running: 'ollama serve'")
    
    try:
        print("📡 Starting server on http://127.0.0.1:8080")
        print("⏳ Please wait for 'Press CTRL+C to quit' message...")
        app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)
        
    except OSError as e:
        if "Address already in use" in str(e):
            print("⚠️  Port 8080 is busy. Trying port 8081...")
            print("🌐 Server URL: http://127.0.0.1:8081")
            app.run(host='127.0.0.1', port=8081, debug=False, use_reloader=False)
        else:
            print(f"❌ Failed to start server: {e}")
            raise