import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from model.model import instance_segmentation_api


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_PATH'] = 4000

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('input_img.png')))
        instance_segmentation_api(img_path=os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png'), out_path=app.config['UPLOAD_FOLDER'])
        input_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
        segmented_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.png')
        return render_template("index.html", input_img=input_img_path, segmented_img=segmented_img_path)
		
if __name__ == '__main__':
   app.run(debug = True)
