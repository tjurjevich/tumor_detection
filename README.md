# Summary  
In recent years, computer vision has consistently been a topic of interest within the machine learning and artificial intelligence communities. One particular application of image classification and detection lives within the healthcare sector. Early detection of cancer can save lives, especially in aggressive forms of brain cancer as demonstrated in this project. Although this model should **not** be used as professional medical interpreter, it provides a structured virtual "playground" for data scientists who are specifically interested in deep learning as it relates to healthcare.  

This project was completed in an effort to further understand standard DevOps practices, paramaterized model training, convolutional neural network architecture, transfer learning & fine-tuning, and source code containerization. 

# End-to-End Setup  
1. Download this repository to your local machine, and set as your current directory.  
```bash
git clone https://github.com/tjurjevich/tumor_detection.git
cd tumor_detection
```

2. Create a local environment in the new repo directory, then activate.  
**Windows**  
```bash
[python/python3] -m venv env
env\Scripts\activate
```

**macOS/Linux**  
```bash
[python/python3] -m venv env
source env/bin/activate
```  

3. Download versioned packages from `requirements.txt` file (this may take a few minutes due to large packages such as tensorflow).  
```bash
pip install -r requirements.txt
```

4. Ensure parameters within `config/params.yaml` are complete and valid (see comments within yaml file for further instructions).  

5. Model training  
There are two model types that are capable of being developed: an entirely custom model, or a transfer learning model. To go the custom model route, you need to ensure all parameters have acceptable values within the `model['custom']` config file. Conversely, all model parameters need to be acceptable within `model['transfer_learning']` to train that model type. 

**Custom model base architecture**  
Rescale  
↓  
Conv2D  
↓  
Pooling (either Max or Average)  
↓  
(additional Conv2D/Pooling blocks if desired...)  
↓  
Flatten    
↓  
Dense  
↓  
(additional Dense layers if desired...)  
↓  
Dropout  
↓  
Classifier (Dense)   

**Transfer learning model base architecture**  
Rescale  
↓  
Frozen Base Model (ResNet50 or DenseNet121)  
↓  
GlobalPooling (either Max or Average)  
↓  
Dense  
↓  
(additional Dense layers if desired...)  
↓  
Dropout  
↓  
Classifier (Dense)  

**To start model training...** 

*custom*
```bash
[python/python3] -m src.training.custom_model
```  

*transfer learning*  
```bash
[python/python3] -m src.training.transfer_learning
```  

Preprocessing and model training logs are output to `logs/` subdirectory. Non-model training information is written to [custom_model or transfer_learning_model]_general_logs_[current date].log. Epoch level metrics and losses are written into [custom_model or transfer_learning_model]_training_logs_[current date].csv.  

**Note**: expect a custom model training to take approximately 15 minutes with current parameters and data volume. Expect a transfer learning model to take approximately 1 hour with current parameters and data volume.  

6. Frontend Dash app  

Once model training is completed successfully, the model is written to disk in the local directory (`./saved_models/`). This model will be used for the front end Dash app for demonstration purposes.  

The current Dash app (`./frontend/app.py`) is a minimal front end application used to display basic information about the test image(s), such as image name, image label, the rendered image, and the model's prediction. The source code can be much improved to add more beneficial and rich content for the task at hand.  

**To launch the Dash app...**  
```bash
[python/python3] -m frontend.app
```  

The UI asks the user to select an image from the following directory: `./data/testing/[tumor/notumor]`. In the event an image outside of these directories is chosen, the image label will simply be listed 'Unknown'.  

**(Optional) To containerize the Dash app...**  

```bash
docker build -t tumor-detection .
docker run -p 8050:8050 tumor-detection 
```  


Then, navigate to `http://localhost:8050` to test out the containerized version of your model directory.  

!(frontend/screenshots/unselected-image.png "UI display when Dash app is initially launched.")  
!(frontend/screenshots/image-selection.png "To make a prediction on an image, click 'Select an Image', and choose image from project directory.")  
!(frontend/screenshots/prediction-output.png "Observe model's prediction just below the rendered image.")  


# Additional information  

- Data used for model training, validation, and testing was downloaded from Kaggle (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). You will notice the original datasets are split into four groups (notumor, glioma, meningioma, pituitary) instead of two (tumor, notumor). Future iterations of this model could work to first identify a tumor, and then classify tumor type if one is initially detected.  

- The project was developed on a 2024 MacBook Pro with M4 Pro chip, 48 GB RAM, and 512GB storage.  