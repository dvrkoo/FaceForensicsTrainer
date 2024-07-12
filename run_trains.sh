# Add more lines for different argument combinations
python3 train.py -e 5 -mp face2face.pt -dr "/home/nick/ff_crops/face2face_crops/"  # the root folder containing 'real' and 'fake' subfolders
python3 train.py -e 5 -mp deepfake.pt -dr "/home/nick/ff_crops/deepfake_crops/"  # the root folder containing 'real' and 'fake' subfolders
python3 train.py -e 5 -mp faceshifter.pt -dr "/home/nick/ff_crops/faceshifter_crops/"  # the root folder containing 'real' and 'fake' subfolders
python3 train.py -e 5 -mp faceswap.pt -dr "/home/nick/ff_crops/faceswap_crops/"  # the root folder containing 'real' and 'fake' subfolders
python3 train.py -e 5 -mp neuraltextures.pt -dr "/home/nick/ff_crops/neuraltextures_crops/"  # the root folder containing 'real' and 'fake' subfolders
