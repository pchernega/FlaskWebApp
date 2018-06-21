import io
import tempfile
import os
from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

def split_data_target(df):
    target = df.target.values
    del df['target']
    data = df.values
    return data, target
    
def get_accuracy(clr, data, target):
    kf = KFold(n_splits=5, shuffle=True)
    accuracy = []
    for tr, ts in kf.split(data):
        clr.fit(data[tr], target[tr])
        accuracy.append(accuracy_score(target[ts], clr.predict(data[ts])))
    return np.mean(accuracy)

def draw_roc_curve(clr, data, target):
    clr.fit(data, target)
    y_score = clr.decision_function(data)
    f, t, _ = roc_curve(target, y_score)
    file = tempfile.NamedTemporaryFile(
        dir='static/images/', delete=False, suffix='.png'
    )
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.plot(f, t)
    canvas = FigureCanvas(fig)
    canvas.print_png(file)
    _, name = os.path.split(file.name)
    return name

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/result', methods=('POST',))
def result():
    file = request.files['file']
    data = io.StringIO(file.read().decode('utf-8'))
    df = pd.read_csv(data)
    content = df.head(10)
    data, target = split_data_target(df)
    lr = LogisticRegression()
    accuracy = get_accuracy(lr, data, target)
    filename = draw_roc_curve(lr, data, target)
    return render_template(
        'result.html', 
        content=content, 
        accuracy=accuracy,
        filename=filename
    )


if __name__ == '__main__':
    app.run(debug=True)
