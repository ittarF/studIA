from config.model_config import asr_model_id, load_wav2vec2
import utils
import torch
from tqdm import tqdm

class ASR:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.processor = load_wav2vec2(asr_model_id[model_name])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def transcribe(self, audio_path, chunk_size=10, show=True):
        if show:
            print('loading audio...')
        speech_array, sampling_rate = utils.load_audio(audio_path)
        input_values = self.processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values
        input_values, = input_values.to(self.device)
        input_values = utils.split_input(input_values, sampling_rate*chunk_size)
        predicted_sentences = []
        if show:
            print('transcribing...')
        for i, input_chunk in enumerate(input_values): 
            # print progress
            if show:
                print(f'chunk {i+1}/{len(input_values)}')
            with torch.no_grad():
                logits = self.model(input_chunk.unsqueeze(0)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            pred = self.processor.batch_decode(predicted_ids)[0]
            if show:
                print(pred)
            predicted_sentences.append(pred)
            
        return " ".join(predicted_sentences)

if __name__ == '__main__':
    asr = ASR("wav2vec2-sm")
    audio_path = "./data/Farmacologia4.m4a"
    transcript = asr.transcribe(audio_path, chunk_size=0, show=True)
    # save transcript to file
    with open('transcript.txt', 'w') as f:
        f.write(transcript)
