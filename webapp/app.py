import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from background_removal.backgroud_removal_dlv3p import backgroundRemovalModelDLV3P
from  background_removal.background_removal_UNET import backgroundRemovalModelUNET
from bokeh.plotting import figure
from bokeh.embed import components
from model_performance.model_performance import get_performance_as_bokeh

m_dlv3p = backgroundRemovalModelDLV3P()
m_unet = backgroundRemovalModelUNET()





app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)


def gen_frames_DL():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            final_frame =  m_dlv3p.pred(frame)
            ret, buffer = cv2.imencode('.jpg', final_frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def gen_frames_Unet():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            final_frame =  m_unet.pred(frame)
            ret, buffer = cv2.imencode('.jpg', final_frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames_DL(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_unet')
def video_unet():
    return  Response(gen_frames_Unet(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/graphs')
def chart_page():
    plot=get_performance_as_bokeh()
    script, div = components(plot)
    return render_template('graphs.html', script=script, div=div)

@app.route('/fineTuning', methods=['GET', 'POST'])
def kaggle_api_page():
    if request.method == 'POST':
        # Retrieve the Kaggle API command from the form data
        api_command = request.form.get('api-command')

        # You can process the API command here or perform any other actions

        # For demonstration purposes, we'll print the command to the console
        print("Kaggle API Command:", api_command)

    return render_template('fineTuning.html')

if __name__ == '__main__':
    app.run(debug=True)