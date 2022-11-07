from flask import Flask,request,render_template
from .utils import classify_image,load_artifacts

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route('/submit',methods=['GET','POST'])
def classify():
    p=''
    img_path=''
    if request.method == 'POST':
        img = request.files['my_image']
        img.filename="latest_img.jpg"
        img_path = "static/" + img.filename	
        img.save(img_path)
        p = classify_image(path=img_path)
        if p=="":
            return render_template("index.html",prediction="SORRY NO FACE DETECTED,PLEASE UPLOAD A DIFFERENT IMAGE",img_path="./static/error.png")
        else:
            return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)