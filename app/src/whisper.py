import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import utils
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class SpeechToTextProcessor:
    def __init__(self, model_id="openai/whisper-large-v3-turbo", device=None, torch_dtype=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id, language="it")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            batch_size=32,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def process_audio(self, audio_path, chunk_size_s=30, save:bool=False):
        # Split audio into chunks
        chunks = utils.split_input_v2(audio_path, chunk_size_s=chunk_size_s)
        
        # Process chunks and get results
        results = self.pipe(chunks, return_timestamps=True)
        
        # Combine the text from all chunks
        full_text = ''.join([el["text"] for el in results])
        
        if save:
            with open('transcript.txt', 'w') as f:
                f.write(full_text)
            print("Transcript saved to transcript.txt")
        
        return full_text

if __name__ == '__main__':
    stt_processor = SpeechToTextProcessor()
    full_text = stt_processor.process_audio("./data/Farmacologia4.m4a", save=True)
    print(full_text)