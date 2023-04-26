from flask import Flask, request, render_template
import os
from gen_hashtags import hashtags


app = Flask(__name__)

# Set the path for uploading files
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hashtags', methods=['POST'])
def upload():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        l = hashtags("static/uploads"+filename)
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return l

if __name__ == '__main__':
    app.run(debug=True)
