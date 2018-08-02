from flask import (     # pylint: disable=import-error
    Flask,
    render_template,
    request,
    url_for,
)  # pylint: disable=import-error
import os
import sys
import pickle
import matplotlib
matplotlib.use('pdf')
import seaborn as sns # pylint: disable=import-error

app = Flask(__name__)

with open('/opt/code/src/web/idx.txt',"rb") as file_to_use:
    idx = pickle.load(file_to_use)

@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def show_index():
    return render_template("index.html", idx=idx)

@app.route("/mat_conf_mieux")
def mat_conf_pas_mal():
    mat_conf_title = "mat_conf"
    image = "/static/mat_conf.png"
    return render_template("mat_conf_mieux.html",image=image,mat_conf_title=mat_conf_title)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

