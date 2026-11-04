from flask import Flask, render_template, request, send_from_directory, redirect, session

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='static')

@app.get('/')
def handle_index():
    return render_template("index.html")

@app.post('/HandleRegister')
def handle_get_():
    print("handling registering")
    data = request.json
    username = data.get('username')
