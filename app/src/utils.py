import librosa

def load_audio(audio_path):
    return librosa.load(audio_path, sr=16_000)

def split_input(input_values, chunk_size=16000):
    length = input_values.shape[-1]
    if length <= chunk_size:
        return [input_values]
    else:
        result = []
        for i in range(0, length, chunk_size):
            result.append(input_values[..., i:i + chunk_size])
        return result