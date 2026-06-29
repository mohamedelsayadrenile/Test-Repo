from faster_whisper import WhisperModel

model = WhisperModel(
    "Systran/faster-whisper-large-v3",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    "5913610109015039607.wav",
    language="ar",
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False
)

for s in segments:
    print(s.text)