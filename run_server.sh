#!/bin/bash

echo "loading anaconda environment 'scoring'..."
source activate scoring
echo "starting nlp-scoring server..."
python server.py
