import whisper 
#오디오 파일-> 텍스트로 변환
whisper_model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]
