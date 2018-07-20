
from flask import Flask, render_template,request
import numpy as np
import keras.models
import re,sys,os
from scipy.misc import  imread, imresize
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import io
from keras.preprocessing import image






APP_ROOT = os.path.dirname(os.path.abspath(__file__))




app = Flask(__name__)

global loaded_model, graph

def init(): 
    json_file = open('flower_recognition.json','r')
    json_file = open('flower_recognition.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("flower_recognition_classifier_weight.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    graph = tf.get_default_graph()

    return loaded_model,graph


loaded_model, graph = init()





def process(img):
    img = imresize(img,(150,150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    with graph.as_default():
        result = loaded_model.predict(img)
        if result[0][0] == 1:
            prediction = 'Daisy'
        else:
            if result[0][1] ==1:
                prediction = 'Dendelion'
            else:
                if result[0][2] ==1:
                    prediction = 'Rose'
                else:
                    if result[0][3] ==1:
                        prediction = 'Sunflower'
                    else:
                        if result[0][4] ==1:
                            prediction = 'Tulip'
                        else:
                            prediction = 'Not flower of these 5 category'

        return prediction
    




@app.route('/')
def index():
    return render_template("index.html")




@app.route('/predict', methods = ['POST'])
def upload():
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img))
    prediction = process(img)

    target = os.path.join(APP_ROOT, 'images')

    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("upload.html", name=prediction)



if __name__ == "__main__":
    app.run(host='127.0.0.1',port=4555, debug=True)

