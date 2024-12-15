from flask import Flask, request
from inference import inference_all

app = Flask(__name__)

@app.post("/")
def inference_api():
    return inference_all(request.form['image_url'])
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)