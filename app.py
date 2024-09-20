import os
from PIL import Image
from flask import Flask, render_template,request, url_for
from utilities.cv2_model import *
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg'}


classesFile = r"flask_project\utilities\coco.names"


def yolo_out(modelConf, modelWeights, classesFile, image):
    net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    inpWidth = 416
    inpHeight = 416
    frame = cv2.imread(image)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)  # pass the image
    net.setInput(blob)
    yolo_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(yolo_layers)
    result_img = post_process(frame, outs, img, classes)
    pil_img = Image.fromarray(result_img)
    processed_filename = 'processed_' + os.path.basename(image)
    processed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    pil_img.save(processed_img_path)
    return processed_filename


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    process_img_url = None
    if request.method == 'POST':
        print(request.files['file'].filename)
        if 'file' not in request.files:
            return 'No file part in the request'
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(filepath)
            result = yolo_out(modelConf, modelWeights, classesFile, filepath)
            process_img_url = url_for('static', filename='uploads/'+result)
    return render_template('index.html', process_img_url=process_img_url)


if __name__ == '__main__':
    app.run(debug=True)
