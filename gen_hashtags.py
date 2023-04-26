import os
import json
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Define hyperparameters
IMG_SIZE = 224
MAX_CAPTION_LEN = 20
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
DROPOUT_RATE = 0.5
EMBEDDING_DIM = 100
LSTM_UNITS = 256

# Load image data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_image_generator = train_datagen.flow_from_directory(
    'new/img_resized',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None
)


captions = []
for filename in train_image_generator.filenames:
    img_id = os.path.splitext(os.path.basename(filename))[0]
    with open(f'new/aptions/newyork/1480884068962566213.txt', 'r') as f:
        caption = f.read().strip()
    captions.append({
        'filename': filename,
        'caption': caption
    })

# Split the caption data into train and validation sets
num_train = int(len(captions) * 0.8)
train_captions = captions[:num_train]
val_captions = captions[num_train:]

# Create vocabulary from captions
word_counts = {}
for caption in train_captions:
    for word in caption['caption'].split():
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

vocab = ['<pad>', '<start>', '<end>', '<unk>'] + list(sorted(word_counts.keys()))
word_to_index = {word: index for index, word in enumerate(vocab)}
index_to_word = {index: word for index, word in enumerate(vocab)}

# Create training data
train_data = []
for caption in train_captions:
    img_id = os.path.splitext(os.path.basename(caption['filename']))[0]
    if img_id in train_image_generator.filenames:
        img = train_image_generator.next()
        if os.path.splitext(os.path.basename(train_image_generator.filenames[train_image_generator.batch_index - 1]))[0] == img_id:
            train_data.append({
                'img': img,
                'caption': caption['caption']
            })

# Create validation data
val_data = []
for caption in val_captions:
    img_id = os.path.splitext(os.path.basename(caption['filename']))[0]
    if img_id in train_image_generator.filenames:
        img = train_image_generator.next()
        if os.path.splitext(os.path.basename(train_image_generator.filenames[train_image_generator.batch_index - 1]))[0] == img_id:
            val_data.append({
                'img': img,
                'caption': caption['caption']
            })

# Define model architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
input_image = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(input_image)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(LSTM_UNITS, activation='relu')(x)
x = RepeatVector(MAX_CAPTION_LEN)(x)

input_caption = Input(shape=(MAX_CAPTION_LEN,))
y = Embedding(input_dim=len(vocab), output_dim=EMBEDDING_DIM, mask_zero=True)(input_caption)
y = LSTM(LSTM_UNITS, return_sequences=True)(y)

z = tf.keras.layers.concatenate([x, y])
z = LSTM(LSTM_UNITS, return_sequences=False)(z)
output = Dense(len(vocab), activation='softmax')(z)

model = Model(inputs=[input_image, input_caption], outputs=output)
model.summary()

# Compile model
optimizer = Adam(lr=LR)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Train model
def generate_batch(data):
    while True:
        for i in range(0, len(data), BATCH_SIZE):
            batch_data = data[i:i+BATCH_SIZE]
            batch_images = np.array([item['img'][0] for item in batch_data])
            batch_captions = [item['caption'] for item in batch_data]
            batch_sequences = [[word_to_index.get(word, word_to_index['<unk>']) for word in caption.split()] for caption in batch_captions]
            batch_sequences = pad_sequences(batch_sequences, maxlen=MAX_CAPTION_LEN, padding='post', truncating='post')
            batch_inputs = {'input_1': batch_images, 'input_2': batch_sequences[:, :-1]}
            batch_outputs = to_categorical(batch_sequences[:, 1:], num_classes=len(vocab))
            yield batch_inputs, batch_outputs

train_generator = generate_batch(train_data)
val_generator = generate_batch(val_data)

history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_data) // BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=val_generator,
                              validation_steps=len(val_data) // BATCH_SIZE)
model.save('model.h5')

def generate_caption(model, image, word_to_index, index_to_word, max_caption_len):
    # Initialize the caption with the start token
    caption = '<start>'

    # Loop until the maximum caption length is reached or an end token is generated
    while True:
        # Convert the current caption to a sequence of integers
        seq = [word_to_index[word] for word in caption.split() if word in word_to_index]
        seq = pad_sequences([seq], maxlen=max_caption_len, padding='post')

        # Generate the next word probabilities
        preds = model.predict([image, seq])[0]
        word_idx = np.argmax(preds)

        # Convert the integer to the corresponding word
        word = index_to_word[word_idx]

        # Stop if the end token is generated or the maximum caption length is reached
        if word == '<end>' or len(caption.split()) >= max_caption_len:
            break

        # Add the predicted word to the caption
        caption += ' ' + word

    return caption

def extract_hashtags(caption):
    # Extract all hashtags using a regular expression
    hashtags = re.findall(r'\#\w+', caption)

    # Remove the '#' symbol from each hashtag
    hashtags = [hashtag[1:] for hashtag in hashtags]

    return hashtags

def hashtags(img_path):
    # Load the trained model
    model = tf.keras.models.load_model('model.h5')

    # Load the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Generate caption for the image
    caption = generate_caption(model, img, word_to_index, index_to_word, max_caption_len=MAX_CAPTION_LEN)

    # Extract hashtags from the caption
    hashtags = extract_hashtags(caption)

    return hashtags