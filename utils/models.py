import tensorflow as tf
from tensorflow.keras.layers import Rescaling, Dense, Conv2D, MaxPooling2D, AvgPooling2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, DenseNet121
from typing import Literal, Iterable


# Class definition for custom model
class CustomTumorClassifier(tf.keras.Model):
    def __init__(self,
        conv_layer_filters: int | Iterable[int] = [16, 32],
        pool_type: Literal['max','avg'] = 'max',
        dense_layer_units: int | Iterable[int] = [128, 64],
        dropout_pct: float = 0.3
    ):
        super().__init__()
        self.single_conv_block = isinstance(conv_layer_filters, int)
        self.single_dense_layer = isinstance(dense_layer_units, int)

        # Define Rescaling layer (constant)
        self.rescale = Rescaling(scale = 1./255)

        # Define Conv/Pooling block
        self.conv_block = []
        if not self.single_conv_block:
            for _, num in enumerate(conv_layer_filters):
                self.conv_block.append(Conv2D(num, (3,3), activation = 'relu'))
                self.conv_block.append(MaxPooling2D() if pool_type == 'max' else AvgPooling2D())
        if self.single_conv_block:
                self.conv_block.append(Conv2D(num, (3,3), activation = 'relu'))
                self.conv_block.append(MaxPooling2D() if pool_type == 'max' else AvgPooling2D())

        # Define Dropout, Flatten
        self.dropout = Dropout(dropout_pct)
        self.flatten = Flatten()

        # Define Dense layers
        if not self.single_dense_layer:
            self.dense_block = []
            for _, units in enumerate(dense_layer_units):
                self.dense_block.append(Dense(units, activation = 'relu'))
        if self.single_dense_layer:
            self.dense_layer = Dense(dense_layer_units, activation = 'relu')

        # Final Dense layer
        self.classifier = Dense(1, activation = 'sigmoid')

    def call(self, inputs, training = False):
        # 1. Rescale inputs (always the first layer)
        x = self.rescale(inputs)

        # 2. Run through user-defined number of Conv/Pooling cycles
        for layer in self.conv_block:
            x = layer(x) 

        # 3. Flatten data
        x = self.flatten(x)

        # 4. Dense layers 
        if self.single_dense_layer:
            x = layer(x)
        if not self.single_dense_layer: 
            for layer in self.dense_block:
                x = layer(x)

        # 5. Dropout layer
        x = self.dropout(x, training = training)

        # 6. Final (categorization) Dense layer
        output = self.classifier(x)

        return output

# Class definition for transfer learning tumor classifier
# Transfer learning classifier method
class TLTumorClassifier(tf.keras.Model):
    def __init__(self, 
        image_height: int = 224, 
        image_width: int = 224, 
        image_channels: int = 3, 
        base_model: Literal['resnet','densenet'] = 'resnet', 
        pool_type: Literal['max','avg'] = 'max',
        dense_layer_units: int | Iterable[int] = 32,
        dropout_pct: float = 0.3
    ):
        super().__init__()
        self.single_dense_layer = isinstance(dense_layer_units, int)

        # Base model assignment
        if base_model == 'resnet':
            self.base_model = ResNet50(
                weights = 'imagenet',
                include_top = False,
                input_shape = (image_height, image_width, image_channels)
            )

        if base_model == 'densenet':
            self.base_model = DenseNet121(
                weights = 'imagenet',
                include_top = False,
                input_shape = (image_height, image_width, image_channels)
            )

        # Base model should not be getting retrained
        self.base_model.trainable = False

        # Standardization layer that resizes to and scales  
        self.rescale = Rescaling(scale = 1./255)

        # Global pooling layer
        self.pool_layer = GlobalMaxPooling2D() if pool_type == 'max' else GlobalAveragePooling2D()

        # Dropout, dense, classifier
        self.dropout = Dropout(dropout_pct)

        if not self.single_dense_layer:
            self.dense_block = []
            for _, units in enumerate(dense_layer_units):
                self.dense_block.append(Dense(units, activation = 'relu'))
        if self.single_dense_layer:
            self.dense_layer = Dense(dense_layer_units, activation = 'relu')
        
        self.classifier = Dense(1, activation = 'sigmoid')
    
    def call(self, inputs, training = False):
        # 1. Rescale
        x = self.rescale(inputs)

        # 2. Base model, either ResNet or DenseNet121
        x = self.base_model(x, training = training)

        # 3. Global pooling layer
        x = self.pool_layer(x)

        # 4. 1+ Dense layers
        if self.single_dense_layer:
            x = layer(x)
        if not self.single_dense_layer: 
            for layer in self.dense_block:
                x = layer(x)
    
        # 5. Dropout
        x = self.dropout(x, training = training)

        # 6. Final Dense layer
        output = self.classifier(x)
        return output

