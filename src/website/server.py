from flask import Flask, render_template, request, jsonify

from model import pass_image
import image_generation

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='static')

@app.get('/')
def handle_index():
    return render_template("index.html")

@app.post('/getPrediction')
def handle_get_():
    print("handling registering")
    data = request.json
    climb_string = data.get('climb_string')
    file_name = image_generation.create_kilterboard_png(climb_string)
    angle = int(data.get("angle"))
    print(f"climb_string: {climb_string}")
    print(f"angle: {angle}")
    #...somehow pass the png through the model
    output = pass_image(file_name, angle)
    response = {"output": output}
    return jsonify(response)

app.run(debug=True)