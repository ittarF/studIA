from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

asr_model_id = {
    "wav2vec2": "jonatasgrosman/wav2vec2-xls-r-1b-italian",
    "wav2vec2-sm": "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
    }

def load_wav2vec2(model_id):
    "output: model, processor"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)    
    return model, processor
