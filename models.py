# SPEECH


def speech_TF_tutorial(name='tensorflow_tutorial_model'):
    # https://www.tensorflow.org/tutorials/audio/simple_audio
    input_shape = ex_spectrograms.shape[1:]
    print('Input shape:', input_shape)
    # num_labels = len(label_names)

    dropout_1 = 0.25
    dropout_2 = 0.5

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization(name='Normalization')
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(
        data=train_spectrogram_ds.map(map_func=lambda audio, spec, label: spec)
    )

    model = models.Sequential([
        layers.Input(shape=input_shape, name='Input'),
        # Downsample the input.
        layers.Resizing(32, 32, crop_to_aspect_ratio=True, name='Resizing'),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu', name='Conv2D_1'),
        layers.Conv2D(64, 3, activation='relu', name='Conv2D_2'),
        layers.MaxPooling2D(name='MaxPooling2D'),
        layers.Dropout(dropout_1, name=f'Dropout_{dropout_1}'),
        layers.Flatten(name='Flatten'),
        layers.Dense(128, activation='relu', name='Dense_1'),
        layers.Dropout(dropout_2, name=f'Dropout_{dropout_2}'),
        layers.Dense(NUM_CLASSES, name='Dense_2'),
    ])

    # Name of the model
    model._name = name

    return model



def medium_example(name='medium_example_model'):
    # https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e&hl=en&gl=it
    input_shape = ex_spectrograms.shape[1:]
    print('Input shape:', input_shape)
    # num_labels = len(label_names)

    dropout_1 = 0.25
    dropout_2 = 0.5

    model = models.Sequential([
        layers.Input(shape=input_shape, name='Input'),
        layers.Resizing(32, 32, crop_to_aspect_ratio=True, name='Resizing'),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      name='Conv2D_1'),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                      name='Conv2D_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D'),
        layers.Dropout(dropout_1, name=f'Dropout_{dropout_1}'),
        layers.Flatten(name='Flatten'),
        layers.Dense(128, activation='relu', name='Dense_1'),
        layers.Dropout(dropout_2, name=f'Dropout_{dropout_2}'),
        layers.Dense(NUM_CLASSES, name='Dense_2')
    ])

    # Name of the model
    model._name = name

    # model.summary()
    return model

# FACE / FLOW
def create_AlexNet_CNN(name='AlexNet_CNN'):
    model = Sequential([
        Conv2D(
            filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(
            filters=256, kernel_size=(5, 5), strides=(1, 1),
            activation='relu', padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(
            filters=384, kernel_size=(3, 3), strides=(1, 1),
            activation='relu', padding="same"
        ),
        BatchNormalization(),
        Conv2D(
            filters=384, kernel_size=(3, 3), strides=(1, 1),
            activation='relu', padding="same"
        ),
        BatchNormalization(),
        Conv2D(
            filters=256, kernel_size=(3, 3), strides=(1, 1),
            activation='relu', padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model._name = name

    return model


def create_Bilotti_CNN(name='Bilotti_CNN'):

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    conv4 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv4 = Conv2D(18, kernel_size=(3, 3), activation='relu')(pool2)
    conv5 = Conv2D(18, kernel_size=(3, 3), activation='relu')(conv4)
    conv6 = Conv2D(18, kernel_size=(3, 3), activation='relu')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(56, kernel_size=(3, 3), activation='relu')(pool3)
    conv8 = Conv2D(56, kernel_size=(3, 3), activation='relu')(conv7)
    conv9 = Conv2D(56, kernel_size=(3, 3), activation='relu')(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)

    conv10 = Conv2D(51, kernel_size=(3, 3), activation='relu')(pool4)
    conv11 = Conv2D(51, kernel_size=(3, 3), activation='relu')(conv10)
    conv12 = Conv2D(51, kernel_size=(3, 3), activation='relu')(conv11)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv12)

    flatten = Flatten()(pool5)

    dense1 = Dense(2048, activation='relu')(flatten)
    drop1 = Dropout(0.25)(dense1)

    dense2 = Dense(1024, activation='relu')(drop1)
    drop2 = Dropout(0.4)(dense2)

    output = Dense(NUM_CLASSES, activation='softmax')(drop2)

    model = Model(inputs, output)

    model._name = name

    return model


def create_VGG16_Imagenet(name='VGG16_Imagenet'):

    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    )
    base_model.trainable = False  # Not trainable weights

    flatten_layer = Flatten()
    dense_layer_1 = Dense(50, activation='relu')
    dense_layer_2 = Dense(20, activation='relu')
    prediction_layer = Dense(NUM_CLASSES, activation='softmax')

    model = Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])

    model._name = name

    return model

def create_EfficientNetB0_Imagenet(name='EfficientNetB0_Imagenet'):

    nb_class = NUM_CLASSES

    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    model = EfficientNetB0(
        include_top=False, input_tensor=inputs, weights="imagenet"
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(nb_class, activation="softmax", name="pred")(x)

    model._name = name
    # Compile
    model = Model(inputs, outputs, name="EfficientNet")

    return model


def create_RESNET50_VGGFACE(name='RESNET50_VGGFACE'):

    nb_class = NUM_CLASSES

    vgg_model = VGGFace(
        model='resnet50', include_top=False, weights='vggface', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    )
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)

    x = Dense(512, activation='relu', name='fc6')(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation='relu', name='fc7')(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation='relu', name='fc8')(x)
    x = Dropout(0.35)(x)

    out = Dense(nb_class, activation='softmax', name='fc9')(x)

    custom_vgg_model = Model(vgg_model.input, out)
    custom_vgg_model._name = name

    return custom_vgg_model


def create_VGG16_VGGFACE(name='VGG16_VGGFACE'):
    nb_class = NUM_CLASSES

    vgg_model = VGGFace(
        include_top=False, weights='vggface', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    )
    vgg_model.trainable = False  # freeze layer
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)

    x = Dense(512, activation='relu', name='fc6')(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation='relu', name='fc7')(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation='relu', name='fc8')(x)
    x = Dropout(0.35)(x)

    out = Dense(nb_class, activation='softmax', name='fc9')(x)

    custom_vgg_model = Model(vgg_model.input, out)
    custom_vgg_model._name = name

    return custom_vgg_model


def create_grigorasi_model(name='vulpe_grigorasi'):

    # 10.1109/ATEE52255.2021.9425073

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    conv1 = Conv2D(256, kernel_size=(3, 3), activation='relu')(inputs)

    conv2 = Conv2D(512, kernel_size=(3, 3), activation='relu')(conv1)
    batch_norm1 = BatchNormalization()(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)
    drop1 = Dropout(0.4)(pool1)

    conv3 = Conv2D(384, kernel_size=(3, 3), activation='relu')(drop1)
    batch_norm2 = BatchNormalization()(conv3)

    pool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)
    drop2 = Dropout(0.4)(pool2)


    conv4 = Conv2D(192, kernel_size=(3, 3), activation='relu')(drop2)
    batch_norm3 = BatchNormalization()(conv4)

    pool3 = MaxPooling2D(pool_size=(2, 2))(batch_norm3)
    drop3 = Dropout(0.4)(pool3)

    conv5 = Conv2D(384, kernel_size=(3, 3), activation='relu')(drop3)
    batch_norm4 = BatchNormalization()(conv5)

    pool4 = MaxPooling2D(pool_size=(2, 2))(batch_norm4)
    drop4 = Dropout(0.4)(pool4)

    flatten = Flatten()(drop4)

    dense1 = Dense(256, activation='relu')(flatten)
    batch_norm5 = BatchNormalization()(dense1)
    drop5 = Dropout(0.3)(batch_norm5)

    output = Dense(NUM_CLASSES, activation='softmax')(drop5)

    model = Model(inputs, output)

    model._name = name

    return model


def create_cnn_model(name='baseline_model'):
    # Source: https://web.archive.org/web/20180712193854/http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf - p.121


    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool2)

    flatten = Flatten()(conv3)

    dense1 = Dense(128, activation='relu')(flatten)

    output = Dense(NUM_CLASSES, activation='softmax')(dense1)

    model = Model(inputs, output)

    model._name = name

    return model


def create_dog_vs_cat_model(name='dog_cat'):
    # https://github.com/LinggarM/Dog-vs-Cat-Classification-with-Transfer-Learning-using-VGG16
    vggmodel = VGG16(
        weights='imagenet', include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    )

    for layers in (vggmodel.layers)[:19]:
        layers.trainable = True

    X = vggmodel.layers[-2].output
    flatten_layer = Flatten()(X)
    # X = Dense(256, activation ='relu')(X)
    # predictions = Dense(1, activation="sigmoid")(flatten_layer)
    predictions = Dense(NUM_CLASSES, activation="softmax")(flatten_layer)
    model_final = Model(vggmodel.input, predictions)

    model_final._name = name

    return model_final