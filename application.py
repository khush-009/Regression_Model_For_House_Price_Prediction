import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from flask import Flask,request,jsonify,render_template

application = Flask(__name__)
app = application

## import linear regression and standard sclaer pickle
regression_model = pickle.load(open('models/regression.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        MedInc = float(request.form.get('MedInc'))
        HouseAge = float(request.form.get('HouseAge'))
        AveRooms = float(request.form.get('AveRooms'))
        AveBedrms = float(request.form.get('AveBedrms'))
        Population = float(request.form.get('Population'))
        AveOccup = float(request.form.get('AveOccup'))
        Latitude = float(request.form.get('Latitude'))
        Longitude = float(request.form.get('Longitude'))

        new_data_scaled =  standard_scaler.transform([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
        result = regression_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
