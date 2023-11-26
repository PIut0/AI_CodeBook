from transformers import pipeline
import pandas as pd
from tqdm import tqdm
pipe = pipeline("audio-classification", model= "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
dic = {
"angry":0,
"fearful":1,
"sad":2,
"disgust":3,
"neutral":4,

"happy":5

}
# , 'calm', ,, 'happy', , 'surprised'
submission = pd.read_csv("/scratch/network/mk8574/audio_sentiment_challenge/baseline_dy/sample_submission.csv")
preds = []
for i in tqdm(submission['id']):
    d = pipe("/scratch/network/mk8574/audio_sentiment_challenge/data/test/"+i+".wav")
    if d[0]["label"]=="surprised" or d[0]["label"]=="calm":
        if d[1]["label"]=="surprised" or d[1]["label"]=="calm":
            preds.append(dic[d[2]["label"]])
        else:
            preds.append(dic[d[1]["label"]])
    else:
        preds.append(dic[d[0]["label"]])
submission['label'] = preds
submission.to_csv("/scratch/network/mk8574/audio_sentiment_challenge/baseline_dy/test_submission.csv", index=False)