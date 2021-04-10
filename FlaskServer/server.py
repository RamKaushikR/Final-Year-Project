import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import speech_recognition as sr
from flask import Flask, request, jsonify
from speakers import add, remove, predict
from models import createModel


app = Flask(__name__)

model = createModel('triplet_conv1d')
threshold = 1.1
print('Model created')

recognizer = sr.Recognizer()
print('Recognizer created')

@app.route('/findSpeaker', methods = ['GET', 'POST'])
def findSpeaker():
    files = request.files['audio_file']
    audio_text = request.form['audio_text']
    dependent = request.form['dependent']

    files.save('Predict/file.wav')
    print('Received file')

    r = predict(audio_text, model, threshold, recognizer, dependent)
    print(r)
    return jsonify(r)
    
@app.route('/addSpeaker', methods = ['GET', 'POST'])
def addSpeaker():
    files = request.files['audio_file']
    speaker = request.form['speaker']
    audio_text = request.form['audio_text']
    dependent = request.form['dependent']

    files.save('Speakers/' + speaker + '.wav')
    
    result = add(speaker, recognizer, audio_text, dependent)
    print(result)

    r = {'result': result}
    return jsonify(r)
    
@app.route('/removeSpeaker', methods = ['GET', 'POST'])
def removeSpeaker():
    speaker = request.form['speaker']
    
    result = remove(speaker)
    print(result)

    r = {'result': result}
    return jsonify(r)
