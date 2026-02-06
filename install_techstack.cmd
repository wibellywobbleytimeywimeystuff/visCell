@echo off
echo Starte Installation der Python-Pakete...

REM Upgrade pip auf die neueste Version
python -m pip install --upgrade pip

REM Installiere die Pakete mit den gewünschten Versionen
python -m pip install customtkinter==5.2.2
python -m pip install tensorflow==2.16.1
python -m pip install opencv-python==4.10.0
python -m pip install numpy==1.26.4

echo Alle Pakete wurden installiert.
pause
