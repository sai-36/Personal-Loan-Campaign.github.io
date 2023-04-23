from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def homedisplay():
    return render_template('home.html')
@app.route('/predictopen', methods=['POST', 'GET'])
def formdisplay():
    return render_template('index.html')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # data0 = int(request.form['full'])
    data1 = int(request.form['id'])
    data2 = int(request.form['age'])
    data3 = int(request.form['experience'])
    data4 = float(request.form['income'])
    data5 = int(request.form['zip'])
    data6 = int(request.form['family'])
    data7 = float(request.form['ccavg'])
    data8 = int(request.form['education'])
    data9 = int(request.form['mortgage'])
    data10 = int(request.form['sa'])
    data11 = int(request.form['cd'])
    data12 = int(request.form['online'])
    data13 = int(request.form['cc'])
    arr = np.array([data2,data4,data6, data7, data8, data9, data10, data11, data12, data13]).reshape(1,-1)
    # arr2 = [data1, data2, data3, data4, data6]
    # arr3 = np.array(arr2, dtype=str)
    # arr4 = [str(i) for i in arr2]
    data=""
    pred = model.predict(arr)
    if pred[0]==1:
        data="Accept"
        print(data)
        return render_template('accept.html', data=pred)
    else:
        data="Reject"
        print(data)
        return render_template('reject.html', data=pred)


if __name__=='__main__':
    app.run(debug=True)
