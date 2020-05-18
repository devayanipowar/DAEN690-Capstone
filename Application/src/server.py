#   basic
from flask import Flask, request, render_template
from flask_basicauth import BasicAuth
from werkzeug.utils import secure_filename
from os import environ, path
import os
import sys
from costplotting import costplotter
from damageassessment import damageinference
import numpy as np
from PIL import Image


#flask stuff
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'a'
PASSWORD = 'a'
app = Flask(__name__)
app.config.from_object(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'a'
app.config['BASIC_AUTH_PASSWORD'] = 'a'
app.config["IMAGE_UPLOADS"] = os.getcwd() + "/static/uploads/"
basic_auth = BasicAuth(app)
#############
print()



#app itself----------------
@app.route('/')
@basic_auth.required
def home():
    return render_template('query.html')


@app.route('/query', methods=['POST', 'GET'])
@basic_auth.required
def submit_query():


    if request.method == 'POST':
        error = None
        resultstatus = False
        try:
            NWLon = request.form['NWLon']
        except KeyError:
            return render_template('query.html', error=KeyError)
        try:
            NWLat = request.form['NWLat']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            SELon = request.form['SELon']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            SELat = request.form['SELat']
        except KeyError:
            return render_template('query.html', error=KeyError)


        #   inputs
        imagebefore = request.files["imagebefore"]
        imagebefore.save(os.path.join(app.config["IMAGE_UPLOADS"], 'before.png'))
        imageafter = request.files["imageafter"]
        imageafter.save(os.path.join(app.config["IMAGE_UPLOADS"], 'after.png'))
        
        #do image analysis
        damageinference()


        distroyedfile = 'static/uploads/after.png'
        predfile = 'static/processed/vizdamage.png'
        # SELat= 38.47967265070876
        # SELon = -122.74118215314147
        # NWLat = 38.48494250901293
        # NWLon = -122.74645201144564
        im = np.array(Image.open(predfile))
        inputs = [distroyedfile,predfile,SELat,SELon,NWLat,NWLon, im]
        out = costplotter(inputs)
        resultstatus = out[0]
        plot_script = out[1]
        plot_div = out[2]
        print(resultstatus)
        return render_template('results.html', resultstatus=resultstatus, plot_script=plot_script, 
                                plot_div=plot_div, error=error)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(environ.get("PORT", 8000)))