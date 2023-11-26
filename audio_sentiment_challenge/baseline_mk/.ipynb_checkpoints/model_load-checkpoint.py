# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")