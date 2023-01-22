oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.
python -m espnet2.bin.tokenize_text \
        --token_type "phn" -f 2- \
        --delimiter "|" \
        --input "/data/Speech-Backbones/Grad-TTS/resources/filelists/jvs/train.txt" \
        --output "/data/Speech-Backbones/Grad-TTS/resources/filelists/jvs/tokens.txt" \
        --non_linguistic_symbols "none" \
        --cleaner "jaconv" \
        --g2p "pyopenjtalk_prosody" \
        --write_vocabulary true \
        --add_symbol "${blank}:0" \
        --add_symbol "${oov}:1" \
        --add_symbol "${sos_eos}:-1"