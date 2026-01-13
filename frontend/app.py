from dash import Dash, html, dcc, Input, Output
from src.utils.inference import process_test_image, load_model, make_prediction
import os
import base64
from PIL import Image 
import io
import numpy as np

# load model. model path is hardcoded, and will need to change for transfer learning model
model = load_model(model_path = 'saved_models/custom_model.keras')

# rearranges various pieces of info in Div format for a selected image
def parse_image_content(content, filename, true_label):
    return html.Div([
        html.H5(filename)
    ]), html.Div([
        html.H5(true_label)
    ]), html.Div([
        html.Img(src=content)
    ])

# determine whether the image is actually a tumor or not
def grab_true_image_label(filename):
    if os.path.exists(os.path.join('./data/testing/tumor', filename)):
        return 'Tumor'
    elif os.path.exists(os.path.join('./data/testing/notumor', filename)):
        return 'Not a Tumor'
    else:
        return 'Unknown'


# initiate app and define layout
app = Dash(__name__)
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Tumor Classification", id = "main-header")
    ]),

    html.Div([
        html.H4("The tumor classification model behind these predictions was constructed entirely from scratch, \
                utilizing numerous blocks constructed of dense, convolutional, and pooling layers. The model was \
                defined using the TensorFlow framework, and trained with multiple callbacks to both minimize overfitting \
                and initiate fine-tuning of model weights. Held-out validation data was used during the model training phase, \
                in which accuracy and recall capped out around 99%+."
                , id = "main-subheader")
    ]),

    html.Div([
        html.Span(html.I('To make a prediction, simply select an image from either the data/tumor or data/notumor directory. \
                         You are allowed to select a JPEG/JPG image outside of these directories, however, the interface cannot \
                         display the true label of the image.'))
    ], id = 'instruction-block'),

    html.Div([
        dcc.Upload(
            [html.Button('Select an Image')], id = 'image-select-button'
        )
    ], id = 'image-select-div'),

    html.Div([
        #image details
        html.Div([
            html.Div([
                html.Div(html.H4('File Name'), id = 'input-filename-header'), 
                html.Div(id = 'input-filename-text')
            ], id = 'input-filename'),
            html.Div(),
            html.Div([
                html.Div(html.H4('True Label'), id = 'input-image-label-header'), 
                html.Div(id = 'input-image-label-text')
            ], id = 'input-image-label'),
        ], id = 'input-info-display'),
        # image
        html.Div(id = 'input-image-display')
    ], id = 'input-div'),

    html.Div([ 
        html.Div(id = 'prediction-text'),
        html.Div(id = 'prediction-pct')
    ], id = 'prediction-div')
    
], id = 'main')

# callback which will update display with selected photo, photo name, and label
@app.callback(
    Output('input-filename-text', 'children'),
    Output('input-image-label-text', 'children'),
    Output('input-image-display', 'children'),
    Input('image-select-button', 'contents'),
    Input('image-select-button', 'filename')
)
def display_selected_image(content, filename):
    if content is None and filename is None:
        return html.Div('N/A'), html.Div('N/A'), None
    else:
        true_label = grab_true_image_label(filename = filename)
        return parse_image_content(content = content, filename = filename, true_label = true_label)
    
# callback which will make actual prediction with selected photo
@app.callback(
    Output('prediction-text','children'),
    Output('prediction-pct', 'children'),
    Output('prediction-text','style'),
    Output('prediction-div','style'),
    Input('image-select-button', 'contents')
)
def predict_label(contents):
    if contents is None:
        return None, None, None, None
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        img_array = np.array(img)
        processed_img_array = process_test_image(method = 'array', img_height = 256, img_width = 256, img_array = img_array)
        text, num = make_prediction(model = model, image = processed_img_array)
        if text == 'Tumor detected':
            return html.Span([text, html.Br(), html.I('(hover here to display tumor probability)')]), f'{num}%', {'color':'red', 'fontWeight':'bold', 'textAlign':'center'}, {'display':'flex', 'textAlign':'center', 'justify-content':'center','align-items':'center','height': '40px', 'marginLeft':'20%', 'marginRight':'20%', 'marginTop':'2%', 'border':'1px solid darkgrey', 'borderRadius':'3px','backgroundColor':'#FFCCCC'}
        else:
            return html.Span([text, html.Br(), html.I('(hover here to display tumor probability)')]), f'{num}%', {'color':'green', 'fontWeight':'bold', 'textAlign':'center'}, {'display':'flex', 'textAlign':'center', 'justify-content':'center','align-items':'center','height': '40px', 'marginLeft':'20%', 'marginRight':'20%', 'marginTop':'2%', 'border':'1px solid darkgrey', 'borderRadius':'3px','backgroundColor':'#90EE90'}

        
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8050)