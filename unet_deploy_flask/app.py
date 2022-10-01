from flask import Flask, render_template, request, send_file, Response
import os
from math import floor
from PIL import Image
import numpy as np
import hashlib
import requests
import shutil

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
savefolder = './static/img/'

def load(checkpoint, model):
    print("==> laoding model")

model = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_example', methods=['GET'])
def download():
    global savefolder
    return send_file(os.path.join(savefolder,'1.png'), as_attachment=True)

#https://stackoverflow.com/questions/6656363/proxying-to-another-web-service-with-flask
def _proxy(*args, **kwargs):
    resp = requests.request(
        method=request.method,
        url=request.url.replace(request.host_url, '127.0.0.1:8001/'),
        headers={key: value for (key, value) in request.headers if key != 'Host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False)

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    response = Response(resp.content, resp.status_code, headers)
    return response

@app.route('/infer', methods=['POST'])
def success():
    global savefolder
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        saveLocation = savefolder + saveLocation
        f.save(saveLocation)

        url = 'http://10.8.0.2:8001/infer'
        #headers = {'Content-type': f.content_type}
        with open(saveLocation,'rb') as f:
            processed_image_req = requests.post(url, files={'file': f})

        output_hash = hashlib.sha1(saveLocation.encode("UTF-8")).hexdigest()[:20]
        output_image = savefolder+output_hash+".jpeg"

        with open(output_image, 'wb') as f:
            f.write(processed_image_req.content)


            #processed_image_req.raw.decode_content = True
            #shutil.copyfileobj(processed_image_req.raw, f)

        return render_template('inference.html' , saveLocation=saveLocation , output_image=output_image)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 8002))
    app.run(host='0.0.0.0', port=port, debug=True)
