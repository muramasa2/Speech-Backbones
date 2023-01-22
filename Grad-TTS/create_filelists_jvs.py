import os
import re
from glob import glob

from sklearn.model_selection import train_test_split

text_paths = sorted(glob("/data/dataset/jvs_ver1/jvs*/*/transcripts_utf8.txt"))
out_path = "/data/Speech-Backbones/Grad-TTS/resources/filelists/jvs"
wav_train = []
text_train = []
wav_valid = []
text_valid = []
wav_test = []
text_test = []

for text_path in text_paths:
    with open(text_path) as f:
        for line in f:
            wav, text = line.split(":")
            wav_path = os.path.join(
                os.path.dirname(text_path), "wav24kHz16bit/" + wav + ".wav"
            )
            if not os.path.exists(wav_path):
                continue

            if re.compile("jvs(([0][0-8][0-9])|[0][9][0])").search(text_path):
                wav_train.append(wav_path)
                text_train.append(text)
            elif re.compile("jvs[0][9][1-5]").search(text_path):
                wav_valid.append(wav_path)
                text_valid.append(text)
            else:
                wav_test.append(wav_path)
                text_test.append(text)

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
