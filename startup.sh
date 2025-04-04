#!/bin/env /bin/bash

# Rulează primul proces
lxterminal --working-directory=/home/andrei/Desktop/ShieldWaveAI/ML_RESTAPI/WebRestApi -e "bash -c 'source webenv/bin/activate && cd shieldwave && python3.10 manage.py runserver; exec bash'"

# Așteaptă ca primul proces să termine înainte de a începe al doilea
lxterminal --working-directory=/home/andrei/Desktop/ShieldWaveAI/ML_RESTAPI/MachineLearning -e "bash -c 'source mlenv/bin/activate && python3.10 full_program.py; exec bash'"
