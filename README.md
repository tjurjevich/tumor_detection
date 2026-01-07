# Summary

# End-to-End Setup

1. Download this repository to your local machine, and set as your current directory.  
```bash
git clone https://github.com/tjurjevich/tumor_detection.git
cd tumor_detection
```

2. Create a local environment in the new repo directory, then activate.  
**Windows**  
```bash
python -m venv env
env\Scripts\activate
```

**macOS/Linux**  
```bash
python -m venv env
source env/bin/activate
```  

3. Download versioned packages from `requirements.txt` file (this may take a few minutes due to large packages such as tensorflow).  
```bash
pip install -r requirements.txt
```

4. Ensure parameters within `config/params.yaml` are complete and valid (see comments within yaml file for further instructions).  

5. Model training  
There are two model types that are capable of being developed: an entirely custom model, or a transfer learning model.  

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