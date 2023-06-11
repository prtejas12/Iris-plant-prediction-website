from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model (1).pkl", "rb"))


@app.route("/")
def index():
    return render_template("input.html")


@app.route("/predict", methods=["POST"])
def predict():
    petal_length = float(request.form.get("pl"))
    petal_width = float(request.form.get("pw"))
    sepal_length = float(request.form.get("sl"))
    sepal_width = float(request.form.get("sw"))
    res = model.predict(np.array([petal_length, petal_width, sepal_length, sepal_width]).reshape(1,4))
    return render_template("input.html",result=str(res));


if __name__ == "__main__":
    app.run(debug=True)
