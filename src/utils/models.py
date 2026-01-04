import tensorflow as tf
from tensorflow.keras.layers import Rescaling, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, DenseNet121
from typing import Literal, Iterable


# Class definition for custom model
@tf.keras.utils.register_keras_serializable()
class CustomTumorClassifier(tf.keras.Model):
    def __init__(self,
        conv_layer_filters: int | Iterable[int] = [16, 32],
        pool_type: Literal['max','avg'] = 'max',
        dense_layer_units: int | Iterable[int] = [128, 64],
        dropout_pct: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.conv_layer_filters = conv_layer_filters
        self.pool_type = pool_type 
        self.dense_layer_units = dense_layer_units 
        self.dropout_pct = dropout_pct
        self.single_conv_block = isinstance(conv_layer_filters, int)
        self.single_dense_layer = isinstance(dense_layer_units, int)

        # Define Rescaling layer (constant)
        self.rescale = Rescaling(scale = 1./255)

        # Define Conv/Pooling block
        self.conv_block = []
        if not self.single_conv_block:
            for _, num in enumerate(conv_layer_filters):
                self.conv_block.append(Conv2D(num, (3,3), activation = 'relu'))
                self.conv_block.append(MaxPooling2D() if pool_type == 'max' else AveragePooling2D())
        if self.single_conv_block:
                self.conv_block.append(Conv2D(num, (3,3), activation = 'relu'))
                self.conv_block.append(MaxPooling2D() if pool_type == 'max' else AveragePooling2D())

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "conv_layer_filters": self.conv_layer_filters,
            "pool_type": self.pool_type,
            "dense_layer_units": self.dense_layer_units,
            "dropout_pct": self.dropout_pct
        })
        return config

# Class definition for transfer learning tumor classifier
# Transfer learning classifier method
@tf.keras.utils.register_keras_serializable()
class TLTumorClassifier(tf.keras.Model):
    def __init__(self, 
        image_height: int = 256, 
        image_width: int = 256, 
        image_channels: int = 3, 
        base_model: Literal['resnet','densenet'] = 'resnet', 
        pool_type: Literal['max','avg'] = 'max',
        dense_layer_units: int | Iterable[int] = 32,
        dropout_pct: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width 
        self.image_channels = image_channels
        self.base_model = self.base_model 
        self.pool_type = pool_type 
        self.dense_layer_units = dense_layer_units 
        self.dropout_pct = dropout_pct
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
            x = self.dense_layer(x)
        if not self.single_dense_layer: 
            for layer in self.dense_block:
                x = layer(x)
    
        # 5. Dropout
        x = self.dropout(x, training = training)

        # 6. Final Dense layer
        output = self.classifier(x)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_height": self.image_height,
            "image_width": self.image_width,
            "image_channels": self.image_channels,
            "base_models": self.base_model,
            "pool_type": self.pool_type,
            "dense_layer_units": self.dense_layer_units,
            "dropout_pct": self.dropout_pct
        })
        return config