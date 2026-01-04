from dash import Dash, html, dcc, Input, Output
from src.utils.inference import process_test_image, load_model, make_prediction
import os
import base64
from PIL import Image 
import io
import numpy as np

model = load_model(model_path = 'saved_models/custom_model.keras')

def parse_image_content(content, filename, true_label):
    return html.Div([
        html.H5(f'File name: {filename}'),
        html.H5(f'True label: {true_label}'),
        html.Img(src=content)
    ])

def grab_true_image_label(filename):
    if os.path.exists(os.path.join('./data/testing/tumor', filename)):
        return 'Tumor'
    elif os.path.exists(os.path.join('./data/testing/notumor', filename)):
        return 'Not a Tumor'
    else:
        return 'Unknown'


app = Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Tumor Classification", id = "main-header")
    ]),

    dcc.Upload([
        html.Button('Select an Image')
    ], id = 'image-select'),

    html.Div(id='image-display'), 

    html.Div(id = 'prediction-text')
])

# callback which will update display with selected photo, photo name, and label
@app.callback(
    Output('image-display', 'children'),
    Input('image-select', 'contents'),
    Input('image-select', 'filename')
)
def display_selected_image(content, filename):
    if content is None and filename is None:
        return []
    else:
        true_label = grab_true_image_label(filename = filename)
        return parse_image_content(content = content, filename = filename, true_label = true_label)
    
# callback which will make actual prediction with selected photo
@app.callback(
    Output('prediction-text','children'),
    Input('image-select', 'contents')
)
def predict_label(contents):
    if contents is None:
        return []
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(io.BytesIO(decoded))
        img_array = np.array(img)
        processed_img_array = process_test_image(method = 'array', img_height = 256, img_width = 256, img_array = img_array)
        text = make_prediction(model = model, image = processed_img_array)
        return text

        


if __name__ == "__main__":
    app.run(debug = True)