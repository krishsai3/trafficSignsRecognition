import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from tensorflow.keras.models import load_model 
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np
from PIL import Image

print("Loading model") 

global model 
model = load_model('C:/Users/saikr/traffic_signs_recognition/trafficTraining/TSR.h5') 

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('C:/Users/saikr/deploy-mlmodel-env/deploy-mlmodel-project/uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    #Step 1
    data = []
    my_image = Image.open(os.path.join('C:/Users/saikr/deploy-mlmodel-env/deploy-mlmodel-project/uploads', filename))
    #Step 2
    my_image_re = my_image.resize((30,30))
    data.append(np.array(my_image_re))
    X_test = np.array(data)
    proba = model.predict(X_test)[0,:]
    ind = np.argsort(proba) 
    print(proba)
    #step 3
    number_to_class = ['Speed limit (20km/h)',
          'Speed limit (30km/h)', 
          'Speed limit (50km/h)',
          'Speed limit (60km/h)', 
          'Speed limit (70km/h)', 
          'Speed limit (80km/h)', 
          'End of speed limit (80km/h)', 
          'Speed limit (100km/h)', 
          'Speed limit (120km/h)', 
          'No passing', 
          'No passing veh over 3.5 tons', 
          'Right-of-way at intersection', 
          'Priority road', 
          'Yield', 
          'Stop', 
          'No vehicles', 
          'Veh > 3.5 tons prohibited', 
          'No entry', 
          'General caution', 
          'Dangerous curve left', 
          'Dangerous curve right', 
          'Double curve', 
          'Bumpy road', 
          'Slippery road', 
          'Road narrows on the right', 
          'Road work', 
          'Traffic signals', 
          'Pedestrians', 
          'Children crossing', 
          'Bicycles crossing', 
          'Beware of ice/snow',
          'Wild animals crossing', 
          'End speed + passing limits', 
          'Turn right ahead', 
          'Turn left ahead', 
          'Ahead only', 
          'Go straight or right', 
          'Go straight or left',
          'Keep right',
          'Keep left',
          'Roundabout mandatory',
          'End of no passing',
          'End of no passing veh > 3.5 tons']
  
    print(ind)
  
    predictions = {
      "class1":number_to_class[ind[42]],
      "class2":number_to_class[ind[41]],
      "class3":number_to_class[ind[40]],
      "prob1":proba[ind[42]],
      "prob2":proba[ind[41]],
      "prob3":proba[ind[40]],
    }
    #Step 4
    return render_template('predict.html', predictions=predictions)
    
if __name__ == "__main__":
  app.run(debug=True)
