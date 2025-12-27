import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template


app=Flask(__name__)

#import pickle files
svc_model=pickle.load(open('svc.pkl','rb'))
scaler_model=pickle.load(open('std_scaler.pkl','rb'))

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/introduction')
def intro():
    return render_template('intro.html')

#route for prediction
@app.route('/predict_heart_disease',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        age=float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        cp = float(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        exang = float(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = float(request.form.get('slope'))
        ca = float(request.form.get('ca'))
        thal = float(request.form.get('thal'))
        
        new_data_scaled=scaler_model.transform([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        result=svc_model.predict(new_data_scaled)
        
        return render_template('home.html',result=result[0])
        
        
    else:
        return render_template('home.html')
        
    
    


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)