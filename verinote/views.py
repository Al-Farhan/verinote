from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from keras.regularizers import l2
from keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.optimizers import Adam
from django.http import HttpResponse
from django.template import RequestContext
from django.template import loader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pytesseract

def preprocess_image(image, image_size):

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Perform segmentation
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Threshold the image
    _, thresh = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize the image
    resized_image = cv2.resize(thresh, image_size)

    return resized_image

# Step 1: Data Collection and Preprocessing
real_currency_dir = r"C:\Users\Farhan\Downloads\verinote\data\Real Currency Dataset"
fake_currency_dir = r"C:\Users\Farhan\Downloads\verinote\data\Fake Currency Dataset"
image_size = (128, 128)
num_classes = 2  # Genuine and Fake

def load_dataset(data_dir, image_size, label):
    data = []
    labels = []

    for image_file in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_file)
        image = cv2.imread(image_path)
        preprocessed_image = preprocess_image(image, image_size)
        data.append(preprocessed_image)
        labels.append(label)

    return np.array(data), np.array(labels)

real_currency_data, real_currency_labels = load_dataset(real_currency_dir, image_size, label=0)
fake_currency_data, fake_currency_labels = load_dataset(fake_currency_dir, image_size, label=1)

data = np.concatenate((real_currency_data, fake_currency_data), axis=0)
labels = np.concatenate((real_currency_labels, fake_currency_labels), axis=0)

#Input from the user
# image_path = input("Enter the path for the currency to be detected: ")  # /content/drive/MyDrive/500-rupee-new-note.jpg

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.9)),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.01)

# Step 3: CNN Model Training
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Expand dimensions to make the input 4-dimensional
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

def predict_currency(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image from '{image_path}'.")
        return "Error"

    if image.size == 0:
        print(f"Loaded image from '{image_path}' is empty.")
        return "Error"

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale image
    resized_image = cv2.resize(gray_image, image_size)
    resized_image = resized_image / 255.0

    prediction = model.predict(np.array([resized_image]))
    label = np.argmax(prediction)

    if label == 0:
        result = 'Genuine'
    else:
        result = 'Fake'

    return result



# img = predict_currency(image_path)
# print(img)

# Load the original image
# original_image = cv2.imread(image_path)

# Grayscale conversion
# gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Edge detection
# edges = cv2.Canny(gray_image, 100, 200)

# Segmentation (thresholding)
# _, thresholded_image = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Create a canvas to display images side by side
# canvas = np.hstack([original_image, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)])

# Close the displayed window manually
cv2.waitKey(0)
cv2.destroyAllWindows()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Make predictions on the test data
test_predictions = model.predict(X_test)
test_predictions = np.argmax(test_predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, average='weighted')
f1 = f1_score(y_test, test_predictions, average='weighted')
confusion = confusion_matrix(y_test, test_predictions)

print(f"Accuracy: {accuracy*100:.3f}%")
print(f"Precision: {precision*100:.3f}%")
print(f"F1 Score: {f1*100:.3f}%")

# print(confusion)

# Define the confusion matrix
confusion_matrix = confusion

# Create a heatmap for the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Predicted Negative', 'Predicted Positive'],
#             yticklabels=['Actual Negative', 'Actual Positive'])

# Add labels and title
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('VeriNote - Confusion Matrix')

# Show the heatmap
# plt.show()

def index(request):
    template = loader.get_template('index.html')
    if request.method == 'POST':
        # Get the user input
        image_path = request.POST.get('image_path')

        # Perform currency prediction
        prediction = predict_currency(image_path)
        messageJ = {"prediction": prediction}

        return HttpResponse(template.render(messageJ, request=request))
    return HttpResponse(template.render({}, request))

class CurrencyDetectionView(APIView):

    def get(self, request, *args, **kwargs):
        # image_path = input("Enter the path for the currency to be detected: ")
        template = loader.get_template('index.html')
        messageJ = {"Result": "img"}
        # return Response(messageJ)
        return HttpResponse(template.render(messageJ, request))