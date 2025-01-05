import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "it"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
SAMPLES = 10

audio_path = "data/sample_short.mp3"
speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)
print(f"Sample loaded: {speech_array.shape[0] / sampling_rate:.2f} seconds")

print('Loading model')
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

