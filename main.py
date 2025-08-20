from flask import Flask, render_template, request, flash, redirect, url_for
import os
import re
import pandas as pd
from werkzeug.utils import secure_filename

from scripts.extract import ext
from scripts.cluster import rec
from scripts.utils import clean_song_name

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__)
app.jinja_env.filters['clean_song_name'] = clean_song_name
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1000 * 1000  # 25MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    data = pd.read_csv('Song_features.csv')
    names = data['song_name'].tolist()
    return render_template("index.html", song_names=names)

@app.route('/hello', methods=['POST'])
def hello():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if not file or file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Check if song exists in dataset
        data = pd.read_csv('Song_features.csv')
        song_exists = filename in data['song_name'].values
        if song_exists:
            recommendations = rec(filename)
            # Remove uploaded file to clean up
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        else:
            # Extract features, cluster, recommend
            feature_file = ext([filename])
            recommendations = rec(feature_file)
        # Use render_template to show recommendations
        return render_template("recommendations.html", song=filename, recommendations=recommendations)
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))

@app.route('/songselect', methods=['POST'])
def songselect():
    song_query = request.form.get('songselection')
    recommendations = rec(song_query)
    return render_template("recommendations.html", song=song_query, recommendations=recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
