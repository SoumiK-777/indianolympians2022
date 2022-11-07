from flask import Flask,request,render_template
import cv2
import numpy as np
import pywt
import base64
import pickle

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route('/submit',methods=['GET','POST'])
def classify():
    file = request.files['my_image']
    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        p = classify_image(img)
        if p=="":
            return render_template("index.html",prediction="SORRY NO FACE DETECTED,PLEASE UPLOAD A DIFFERENT IMAGE",image_to_show="./static/error.png")
        else:
            image_content = cv2.imencode('.jpg', img)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            return render_template('index.html',image_to_show=to_send,prediction=p)
    except:
        return render_template("index.html",prediction="SORRY NO FACE DETECTED,PLEASE UPLOAD A DIFFERENT IMAGE",image_to_show="./static/error.png")

__class_name_to_number = {}
__class_number_to_name = {}
__model=None

def classify_image(img):
    imgs=get_cropped_image(img)
    result=""
    final_img_array=""
    for img in imgs:
        sc_rw_img=cv2.resize(img,(32,32))
        img_har=w2d(sc_rw_img,'db1',5)
        sc_img_har=cv2.resize(img_har,(32,32))
        combined_img=np.vstack((sc_rw_img.reshape((32*32*3,1)),sc_img_har.reshape(32*32,1)))

        len_img_array=32*32*3+32*32

        final_img_array=combined_img.reshape(1,len_img_array).astype(float)
    try:
        result=__class_number_to_name[int(__model.predict(final_img_array)[0])]
    except:
        pass
    return result

def get_cropped_image(img):
    face_cascade=cv2.CascadeClassifier("./haar-cascade-files-master/haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier("./haar-cascade-files-master/haarcascade_eye.xml")

    # img=cv2.imread(img_path)

    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces=face_cascade.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=5)
    cropped_faces=[]
    for(x,y,w,h) in faces:
        roi_gray=img_gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            cropped_faces.append(roi_color)
    return cropped_faces

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def load_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open("./artifacts/names.pkl","rb") as f:
        __class_name_to_number=pickle.load(f)
        __class_number_to_name={v:k for k,v in __class_name_to_number.items()}

    with open("./artifacts/model_pickle.pkl","rb") as f:
        __model=pickle.load(f)


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)