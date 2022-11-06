from flask import Flask,request,jsonify
import utils

app = Flask(__name__)

@app.route('/classify-image',methods=['GET','POST'])
def classify_image():
    img_data=request.form['img_data']
    response=jsonify(utils.classify_image(img_data))
    response.headers.add('Access-Control-Allow-Origin','*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Image Classification")
    utils.load_artifacts()
    app.run(port=5000, debug=True)