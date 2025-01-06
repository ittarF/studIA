import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

LANG_ID = "it"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"

audio_path = "examples/sample.mp3"
speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)
print(f"Sample loaded: {speech_array.shape[0] / sampling_rate:.2f} seconds")

print('Loading model')
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


import time
input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values
model, input_values = model.to(device), input_values.to(device)
# calculate time
start_time = time.time()
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)
end_time = time.time()
for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Prediction:", predicted_sentence)
    print("-" * 100)
    print(f"Time elapsed: {end_time - start_time:.2f} seconds for a {speech_array.shape[0] / sampling_rate:.2f} seconds audio")
