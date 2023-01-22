import os
import re
from glob import glob

from sklearn.model_selection import train_test_split

# dataset / VCTK - Corpus / txt / p376 / p376_001.txt
text_paths = sorted(glob("/data/dataset/VCTK-Corpus/txt/p*/*.txt"))
out_path = "/data/Speech-Backbones/Grad-TTS/resources/filelists/vctk"
wav_train = []
text_train = []
spkid_train = []
wav_valid = []
text_valid = []
spkid_valid = []

wav_test = []
text_test = []
spkid_test = []

text_train_paths = []
text_valid_paths = []
text_test_paths = []

test_speakers = [225, 234, 238, 245, 248, 261, 294, 302, 326, 335, 347]

for text_path in text_paths:
    speaker_id = re.findall("p([0-9]{3})", text_path)
    if int(speaker_id[0]) in test_speakers:
        text_test_paths.append(text_path)
    else:
        text_train_paths.append(text_path)


import numpy as np

unique_train_spkid = np.unique(re.findall("p([0-9]{3})", " ".join(text_train_paths)))
speaker_to_id = dict(zip(unique_train_spkid, np.arange(len(unique_train_spkid))))
# print(speaker_to_id)

# print(len(np.unique(re.findall("p([0-9]{3})", " ".join(text_train_paths)))))

text_train_paths, text_valid_paths = train_test_split(
    text_train_paths, test_size=0.05, shuffle=True
)


# print(len(text_train_paths))
# print(len(text_valid_paths))
# print(len(text_test_paths))

for text_path in text_train_paths:
    with open(text_path) as f:
        text = f.read().rstrip()
        wav = os.path.basename(text_path).split(".")[0]
        speaker_id = re.findall("p([0-9]{3})", text_path)
        wav_path = os.path.join(
            os.path.dirname(text_path).replace("txt", "wav48"), wav + ".wav"
        )
        if not os.path.exists(wav_path):
            continue

        wav_train.append(wav_path)
        text_train.append(text)
        spkid_train.append(speaker_id[0])

for text_path in text_valid_paths:
    with open(text_path) as f:
        text = f.read().rstrip()
        wav = os.path.basename(text_path).split(".")[0]
        speaker_id = re.findall("p([0-9]{3})", text_path)

        wav_path = os.path.join(
            os.path.dirname(text_path).replace("txt", "wav48"), wav + ".wav"
        )
        if not os.path.exists(wav_path):
            continue

        wav_valid.append(wav_path)
        text_valid.append(text)
        spkid_valid.append(speaker_id[0])

for text_path in text_test_paths:
    with open(text_path) as f:
        text = f.read().rstrip()
        wav = os.path.basename(text_path).split(".")[0]
        speaker_id = re.findall("p([0-9]{3})", text_path)

        wav_path = os.path.join(
            os.path.dirname(text_path).replace("txt", "wav48"), wav + ".wav"
        )
        if not os.path.exists(wav_path):
            continue

        wav_test.append(wav_path)
        text_test.append(text)
        spkid_test.append(speaker_id[0])

os.makedirs(out_path, exist_ok=True)
with open(os.path.join(out_path, "train.txt"), "w") as out:
    for i in range(len(wav_train)):
        out.write(
            wav_train[i]
            + "|"
            + text_train[i]
            + "|"
            + str(speaker_to_id[str(spkid_train[i])])
            + "\n"
        )

with open(os.path.join(out_path, "valid.txt"), "w") as out:
    for i in range(len(wav_valid)):
        out.write(
            wav_valid[i]
            + "|"
            + text_valid[i]
            + "|"
            + str(speaker_to_id[str(spkid_valid[i])])
            + "\n"
        )

with open(os.path.join(out_path, "test.txt"), "w") as out:
    for i in range(len(wav_test)):
        out.write(wav_test[i] + "|" + text_test[i] + "|" + spkid_test[i] + "\n")
