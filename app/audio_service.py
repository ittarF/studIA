import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def split_input(input_values, chunk_size=16000):
    length = input_values.shape[-1]
    if length <= chunk_size:
        return [input_values]
    else:
        result = []
        for i in range(0, length, chunk_size):
            result.append(input_values[..., i:i + chunk_size])
        return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

def transcribe_audio(audio_path, model, processor, chunk_size=16000):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)
    input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)
    model = model.to(device)
    input_values = split_input(input_values, chunk_size)
    predicted_sentences = []
    for i, input_chunk in enumerate(input_values):
        print(f"Transcribing chunk {i + 1} out of {len(input_values)}")
        with torch.no_grad():
            logits = model(input_chunk).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences.append(processor.batch_decode(predicted_ids)[0])
    return " ".join(predicted_sentences)

if __name__ == '__main__':
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
    #MODEL_ID = "dbdmg/wav2vec2-xls-r-1b-italian-robust" # meh
    #MODEL_ID = "jonatasgrosman/wav2vec2-xls-r-1b-italian"


    audio_path = "/Users/giofratti/nerd/studIA/examples/sample_short.mp3"
    speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)
    print(f"Sample loaded: {speech_array.shape[0] / sampling_rate:.2f} seconds")

    print('Loading model')
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    print('Model loaded')
    
    print(transcribe_audio(audio_path, model, processor, chunk_size=16000*10))