import mimetypes
import torchaudio
import os
import logging
import wave
import subprocess
import scipy.io.wavfile
import scipy.signal
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image
from transformers import pipeline
from pydub import AudioSegment
import shutil
import tempfile
import base64
from io import BytesIO
import stat

# FFmpeg configuration
ffmpeg_bin_path = "C:\\Users\\Admin\\Downloads\\deepfake detector\\deepfake detector\\backend\\ffmpeg-2025-06-08-git-5fea5e3e11-essentials_build\\ffmpeg-2025-06-08-git-5fea5e3e11-essentials_build\\bin"
ffmpeg_path = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
os.environ["PATH"] += os.pathsep + ffmpeg_bin_path
AudioSegment.ffmpeg = ffmpeg_path

# Additional environment variables for torchaudio
os.environ["FFMPEG_PATH"] = ffmpeg_path  # Explicitly set for torchaudio
os.environ["TORCHAUDIO_USE_FFMPEG"] = "1"  # Force torchaudio to use FFmpeg

# Verify FFmpeg availability
if not os.path.exists(ffmpeg_path):
    logger.error(f"FFmpeg executable not found at {ffmpeg_path}")
    raise SystemExit(f"FFmpeg executable not found at {ffmpeg_path}. Please verify the path.")
try:
    result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
    logger.info(f"FFmpeg found: {result.stdout.splitlines()[0]}")
except FileNotFoundError:
    logger.error(f"FFmpeg not found at {ffmpeg_path}. Please ensure FFmpeg is installed and accessible.")
    raise SystemExit(f"FFmpeg not found at {ffmpeg_path}. Please ensure FFmpeg is installed and accessible.")
except subprocess.CalledProcessError as e:
    logger.error(f"FFmpeg check failed: {e.stderr}")
    raise SystemExit(f"FFmpeg check failed: {e.stderr}")

# Verify torchaudio can load a test WAV file
try:
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run([ffmpeg_path, "-f", "lavfi", "-i", "sine=frequency=1000:duration=1", "-ar", "16000", "-ac", "1", temp_wav], capture_output=True, text=True, check=True)
    waveform, sample_rate = torchaudio.load(temp_wav)
    logger.info(f"torchaudio successfully loaded test WAV file: shape={waveform.shape}, sample_rate={sample_rate}")
    os.remove(temp_wav)
except Exception as e:
    logger.error(f"torchaudio failed to load test WAV file: {str(e)}")
    raise SystemExit(f"torchaudio failed to load test WAV file: {str(e)}")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
frame_output_folder = "frames"
face_output_folder = "faces"

deepfake_model_path = os.getenv("DEEPFAKE_MODEL_PATH", "1.h5")

# Load video model
try:
    video_model = tf.keras.models.load_model(deepfake_model_path, compile=False)
    logger.info("Video model loaded successfully")
except Exception as e:
    logger.error(f"Error loading video model: {str(e)}")
    raise SystemExit(f"Failed to load video model: {str(e)}")

# Load audio classification pipeline
try:
    audio_classifier = pipeline("audio-classification", model="as1605/Deepfake-audio-detection-V2")
    logger.info("Deepfake-audio-detection-V2 pipeline loaded successfully")
except Exception as e:
    logger.error(f"Error loading audio pipeline: {str(e)}")
    raise SystemExit(f"Failed to load audio pipeline: {str(e)}")

IMG_WIDTH, IMG_HEIGHT = 224, 224

def extract_frames(video_path):
    os.makedirs(frame_output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(7):
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(frame_output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
    cap.release()
    return frames

def detect_faces():
    os.makedirs(face_output_folder, exist_ok=True)
    detector = MTCNN()
    face_images = []
    face_paths = []
    for file_name in os.listdir(frame_output_folder):
        frame_path = os.path.join(frame_output_folder, file_name)
        image = cv2.imread(frame_path)
        faces = detector.detect_faces(image)
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            face_img = image[max(0, y):y+h, max(0, x):x+w]
            face_path = os.path.join(face_output_folder, f"{file_name}_face_{i}.jpg")
            cv2.imwrite(face_path, face_img)
            face_images.append(face_img)
            face_paths.append(face_path)
    return face_paths

def classify_faces():
    face_files = os.listdir(face_output_folder)
    face_images = []
    face_data = []
    
    for file_name in face_files:
        img_path = os.path.join(face_output_folder, file_name)
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = tf.image.convert_image_dtype(img, tf.float32)
            face_images.append(img)
            face_data.append((file_name, img_path))
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            continue
    
    if not face_images:
        logger.warning("No faces detected for classification")
        return [], [], "No faces detected in the video"
    
    try:
        face_images = np.array(face_images)
        predictions = video_model.predict(face_images)
        labels = ['Fake' if p[0] <= 0.5 else 'Real' for p in predictions]
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
        raise
    
    fake_faces_base64 = []
    for (file_name, img_path), label in zip(face_data, labels):
        if label == 'Fake':
            try:
                with open(img_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                    fake_faces_base64.append({
                        "filename": file_name,
                        "image": f"data:image/jpeg;base64,{img_base64}"
                    })
            except Exception as e:
                logger.error(f"Error encoding image {img_path}: {str(e)}")
                continue
    
    return labels, fake_faces_base64, None

AudioSegment.ffmpeg = "C://Users//Admin//Downloads//deepfake detector//deepfake detector//backend//ffmpeg-2025-06-08-git-5fea5e3e11-essentials_build//ffmpeg-2025-06-08-git-5fea5e3e11-essentials_build//bin//ffmpeg.exe"

def preprocess_audio(file_path, original_filename):
    _, ext = os.path.splitext(original_filename.lower())
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if ext not in supported_formats:
        logger.error(f"Unsupported audio format: {ext}")
        raise ValueError(f"Unsupported audio format: {ext}")
    
    if not os.path.exists(file_path):
        logger.error(f"Temporary file {file_path} does not exist")
        raise FileNotFoundError(f"Temporary file {file_path} does not exist")
    file_size = os.path.getsize(file_path)
    logger.info(f"Temporary file {file_path} size: {file_size} bytes")
    if file_size < 1024:
        logger.error(f"Temporary file {file_path} is too small")
        raise ValueError(f"Temporary file {file_path} is too small")
    
    try:
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        logger.info(f"Set permissions for {file_path}")
    except PermissionError as e:
        logger.error(f"Permission error setting chmod for {file_path}: {str(e)}")
        raise
    
    if ext.lower() != '.wav':
        converted_path = file_path.rsplit(".", 1)[0] + "_pcm.wav"
        ffmpeg_path = "C://Users//Admin//Downloads//deepfake detector//deepfake detector//backend//ffmpeg-2025-06-08-git-5fea5e3e11-essentials_build//ffmpeg-2025-06-08-git-5fea5e3e11-essentials_build//bin//ffmpeg.exe"
        try:
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-i", file_path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    "-c:a", "pcm_s16le",
                    converted_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"FFmpeg converted to PCM WAV: {converted_path}, size: {os.path.getsize(converted_path)} bytes")
            logger.debug(f"FFmpeg stdout: {result.stdout}")
            file_path = converted_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise ValueError(f"FFmpeg conversion failed: {e.stderr}")
    
    try:
        with wave.open(file_path, 'rb') as wav_file:
            logger.info(f"WAV file validated: channels={wav_file.getnchannels()}, rate={wav_file.getframerate()}, format={wav_file.getsampwidth()}")
    except wave.Error as e:
        logger.error(f"Invalid WAV header in {file_path}: {str(e)}")
        raise ValueError(f"Invalid WAV header: {str(e)}")
    
    try:
        sample_rate, audio = scipy.io.wavfile.read(file_path)
        logger.info(f"Audio loaded: samples={audio.shape[0]}, rate={sample_rate}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != 16000:
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = scipy.signal.resample(audio, num_samples)
            sample_rate = 16000
        logger.info(f"Processed audio: samples={audio.shape[0]}, rate={sample_rate}")
        return file_path  # Return file path for pipeline
    except Exception as e:
        logger.error(f"Error loading WAV with scipy: {str(e)}")
        raise ValueError(f"Error loading WAV: {str(e)}")

def classify_audio(file_path, original_filename):
    try:
        # Preprocess audio to ensure WAV format and 16kHz
        processed_file_path = preprocess_audio(file_path, original_filename)
        
        # Classify using pipeline
        results = audio_classifier(processed_file_path)
        
        # Find the highest-scoring label
        top_result = max(results, key=lambda x: x['score'])
        label = top_result['label'].capitalize()  # Convert REAL/FAKE to Real/Fake
        confidence = top_result['score']
        
        logger.info(f"Audio classification: {label} with confidence {confidence:.2f}")
        return label, confidence
    except Exception as e:
        logger.error(f"Error classifying audio: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

@app.post("/detect/video")
async def detect_video_deepfake(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        logger.info(f"Processing video: {tmp_path}")
        extract_frames(tmp_path)
        detect_faces()
        face_labels, fake_faces, error_message = classify_faces()
        
        if error_message:
            return JSONResponse({
                "result": "Unknown",
                "real_faces": 0,
                "fake_faces": 0,
                "fake_face_images": [],
                "confidence": 0,
                "error": error_message
            })
        
        fake_count = face_labels.count('Fake')
        real_count = face_labels.count('Real')
        result = "Fake" if fake_count > real_count else "Real"
        
        return JSONResponse({
            "result": result,
            "real_faces": real_count,
            "fake_faces": fake_count,
            "fake_face_images": fake_faces,
            "confidence": max(fake_count, real_count) / (fake_count + real_count) if (fake_count + real_count) > 0 else 0
        })
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/detect/audio")
async def detect_audio_deepfake(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    MAX_FILE_SIZE = 100 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 100MB")
    
    tmp_path = None
    converted_path = None
    temp_dir = "C://Users//Admin//Downloads//deepfake detector//deepfake detector//temp"
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Using temp directory: {temp_dir}")
        
        logger.info(f"Uploaded file: {file.filename}, content_type: {file.content_type}, size: {len(content)} bytes")
        
        mime_type = file.content_type
        ext = mimetypes.guess_extension(mime_type) or '.wav'
        if not ext:
            ext = os.path.splitext(file.filename)[1] if file.filename else '.wav'
        logger.info(f"Assigned extension: {ext}")
        
        if len(content) < 1024:
            logger.error("Uploaded file is too small to be a valid audio file")
            raise HTTPException(status_code=400, detail="Uploaded file is too small to be a valid audio file")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=temp_dir) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
            logger.info(f"Temporary file {tmp_path} created, size: {os.path.getsize(tmp_path)} bytes")
        
        result, confidence = classify_audio(tmp_path, file.filename or "unknown" + ext)
        
        return JSONResponse({
            "result": result,
            "confidence": confidence
        })
    except PermissionError as e:
        logger.error(f"Permission error accessing {tmp_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Permission error: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            logger.info(f"Temporary file {tmp_path} not deleted for debugging")
        if converted_path and os.path.exists(converted_path):
            logger.info(f"Converted file {converted_path} not deleted for debugging")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)