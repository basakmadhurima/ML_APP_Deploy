import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.preprocessing import StandardScaler,scale


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features1 = np.asarray(features)
    final_features1 = final_features1.reshape(1,-1)  
    prediction = model.predict(final_features1)
    print("final features",final_features1)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)

    if output == 0:
        return render_template('index.html', prediction_text='The patient seems to have a heart disease')
    else:
         return render_template('index.html', prediction_text='The patient seems to have no heart disease ')
     
   
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)