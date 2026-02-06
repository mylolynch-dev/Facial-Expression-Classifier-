from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam


class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # you have to initialize self.model to a keras model
        self.model = Sequential([
            # Rescale input images:
            Rescaling(1./255, input_shape=input_shape),

            # Convolutional layers and maxpooling:
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Dropout layer to reduce overfitting:
            layers.Dropout(0.5),

            # Flatten layer:
            layers.Flatten(),

            # Fully connected layer:
            layers.Dense(128, activation='relu'),

            # Softmax activation:
            layers.Dense(categories_count, activation='softmax')
        ])

        # Check number of parameters:
        print("MODEL SUMMARY \n")
        self.model.summary()


    def _compile_model(self):
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy',]
        )
