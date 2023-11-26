# Use a pipeline as a high-level helper
#from transformers import pipeline

#pipe = pipeline("audio-classification", model="Rajaram1996/Hubert_emotion")

# Load model directly
from transformers import AutoProcessor, HubertForSpeechClassification

processor = AutoProcessor.from_pretrained("Rajaram1996/Hubert_emotion")
model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")