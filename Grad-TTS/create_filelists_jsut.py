import os

from sklearn.model_selection import train_test_split

text_path = "/data/dataset/jsut_ver1.1/basic5000/transcript_utf8.txt"
out_path = "/data/Speech-Backbones/Grad-TTS/resources/filelists/jsut"
wav_paths = []
texts = []

with open(text_path) as f:
    for line in f:
        wav, text = line.split(":")
        wav_path = os.path.join(os.path.dirname(text_path), "wav/" + wav + ".wav")
        wav_paths.append(wav_path)
        texts.append(text)

wav_train, wav_test, text_train, text_test = train_test_split(
    wav_paths, texts, test_size=0.05, shuffle=True
)
wav_test, wav_valid, text_test, text_valid = train_test_split(
    wav_test, text_test, test_size=0.2, shuffle=True
)

os.makedirs(out_path, exist_ok=True)
with open(os.path.join(out_path, "train.txt"), "w") as out:
    for i in range(len(wav_train)):
        out.write(wav_train[i] + "|" + text_train[i])

with open(os.path.join(out_path, "valid.txt"), "w") as out:
    for i in range(len(wav_valid)):
        out.write(wav_valid[i] + "|" + text_valid[i])

with open(os.path.join(out_path, "test.txt"), "w") as out:
    for i in range(len(wav_test)):
        out.write(wav_test[i] + "|" + text_test[i])
