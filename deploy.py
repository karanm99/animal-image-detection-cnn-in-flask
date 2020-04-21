from flask import Flask
from flask import request, render_template
from CNN_test import predict_img

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route('/predict_animal', methods=['POST'])
def test_image():

    print("Posted file: {}".format(request.files['File']))

    f = open('temp.jpg', 'wb')
    f.write(request.files['File'].read())
    f.close()

    prediction =predict_img(img_path='temp.jpg')
    # pdb.set_trace()
    return ({'response': prediction}, 200)


if __name__ == "__main__":
    app.run()