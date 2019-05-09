"""Flask app for titanic survival prediction."""
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    return "TODO"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
