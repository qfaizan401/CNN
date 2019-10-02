import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
model = load_model('swift_and_wagonR_final.h5')
def perdiction (file):
    img=image.load_img(file,
                  target_size=(150,150))
    img=image.img_to_array(img)
    img=img.reshape((1,)+img.shape)
    img=img/255
    prediction=model.predict(img)
    return round(float(prediction))
file=input("Enter the file"'\n')
perdiction=perdiction(file)
if perdiction<1:
    print("The given image is of Swift")
else:
    print("The given image is of WagonR")

img = Image.open(file)
img.show()