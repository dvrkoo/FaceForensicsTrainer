ATTENZIONE: sarebbe consigliabile eseguire il programma su un computer con gpu dedicata possibilmente Nvidia, il programma gira anche su cpu ma con scarse prestazioni

1) Installare python3 (qualora non fosse installato) da https://www.python.org/downloads/, l'installer dovrebbe comprendere anche pip
ATTENZIONE:non installare l'ultima versione di python in quanto appena uscita, potrebbe causare problemi con le librerie, l'applicazione è stata testata con python3.11 e 3.12
una volta installato python bisogna creare un venv (virtual environment) per installare le librerie necessarie senza intaccare l'ambiente di sistema
si può creare un venv con il seguente comando da terminale nella root del progetto:
python3.11 -m venv faceforensics_env
successivamente si attiva con il comando:
source faceforensics_env/bin/activate
2) le librerie necessarie per il player sono installabili con "pip3 install nome_libreria" 
le librerie da installare sono:
pip3 install opencv-python
pip3 install numpy
pip3 install pyqt5
pip3 install dlib
pip3 install torch torchvision torchaudio
pip3 install pyinstaller
3) prima di generare l'installer consiglio di eseguire il programma eseguendo da terminale python3 player.py dalla root del progetto per verificare che il programma venga eseguito correttamente.
Al primo avvio potrebbe essere necessario dargli un po' di tempo affinche il programma venga eseguito.
per generare un eseguibile bisognerà digitare in un terminale sempre nella cartella FaceForensicsTrainer:
pyinstaller player.spec 
verranno generate 2 cartelle: dist and build, l'eseguibile si troverà nella cartella dist

QUALORA NON FUNZIONASSE L'INSTALLER:
L'applicazione può essere eseguita aprendo un terminale nella cartella del progetto tramite:

python3 player.py




