import streamlit as st
import cv2
import numpy as np
import os
import time
import glob
import os
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import pip

def install(package):
    pip.main(['install', package])

# Example
if __name__ == '__main__':
    install('gTTS')
from gtts import gTTS


try:
    os.mkdir("temp")
except:
    pass
    
st.title("Proyecto 2")
st.subheader("Nicolas Gonzalez y David García")
st.write('En este proyecto la idea es que el usuario pueda aprender lenguaje de señas colombiano, '
        'por ahora solo es posible de reconocer las siguientes palabras'
        'Cabello, cabeza, ceja, dientes, frente, menton, nariz, oreja. '
        'Ademas lo juntamos con el texto a audio para que te diga en voz alta')

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      texto_rep = "Cabello"
      st.header('Cabello, con Probabilidad: '+str( prediction[0][0]) )
    if prediction[0][1]>0.5:
      texto_rep = "Cabeza"
      st.header('Cabeza, con Probabilidad: '+str( prediction[0][1]))
    if prediction[0][2]>0.5:
      texto_rep = "Ceja"
      st.header('Ceja, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][3]>0.5:
      texto_rep = "Dientes"
      st.header('Dientes, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][4]>0.5:
      texto_rep = "Frente"
      st.header('Frente, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][5]>0.5:
      texto_rep = "Menton"
      st.header('Menton, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][6]>0.5:
      texto_rep = "Nariz"
      st.header('Nariz, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][7]>0.5:
      texto_rep = "Oreja"
      st.header('Oreja, con Probabilidad: '+str( prediction[0][2]))

tld = "es"

def text_to_speech(text, tld):
    tts = gTTS(text, "es", tld, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, text

if st.button("Reproducir"):
    result, output_text = text_to_speech(texto_rep, tld)
    audio_file = open(f"temp/{result}.mp3", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Tu audio:")
    st.audio(audio_bytes, format="audio/mp3", start_time=0)

    st.markdown(f"## Texto en audio:")
    st.write(f" {output_text}")

def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)

remove_files(7)







