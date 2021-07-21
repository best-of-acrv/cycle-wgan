#!/bin/bash

#Download from Google Drive - https://drive.google.com/file/d/1cJ-Hl5F9LOn4l-53vhu3-zfv0in2ahYI/view
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cJ-Hl5F9LOn4l-53vhu3-zfv0in2ahYI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cJ-Hl5F9LOn4l-53vhu3-zfv0in2ahYI" -O data/datasets.zip

#Unzip contents to ./data
echo "Unzipping contents..."
unzip data/datasets.zip -d data/