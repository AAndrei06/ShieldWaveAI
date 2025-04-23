#!/bin/env /bin/bash

# Rulează primul proces
lxterminal --working-directory=/home/andrei/Desktop/ShieldWaveAI/ShieldWaveAI/WebRestApi -e "bash -c 'source webenv/bin/activate && cd shieldwave && python3 manage.py runserver; exec bash'" &

# Așteaptă ca primul proces să termine înainte de a începe al doilea
lxterminal --working-directory=/home/andrei/Desktop/ShieldWaveAI/ShieldWaveAI/MachineLearning -e "bash -c 'source mlenv/bin/activate && python3 full_program.py; exec bash'" &
