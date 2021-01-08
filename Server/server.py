import numpy as np
from flask import Flask, request, jsonify
from speakers import add, remove, predict
from model import createModel


app = Flask(__name__)

model = createModel('triplet_conv1d')

@app.route('/findSpeaker', methods = ['GET', 'POST'])
def findSpeaker():
	result = 'Not found'

    files = request.files['audio_file']
#    speaker = request.form['speaker']
#    print('Speaker to be identified is ' + speaker)

    files.save('Predict/file.wav')
    print('Received file')

    result = predict(model)
    print(result)

    r = {'result': result}
    return jsonify(r)
    
@app.route('/addSpeaker', methods = ['GET', 'POST'])
def addSpeaker():
    file = request.files['audio_file']
    speaker = request.form['speaker']

    file1.save('Speakers/' + speaker + '.wav')
    
    result = add(speaker)
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
