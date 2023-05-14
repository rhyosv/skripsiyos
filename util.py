import tensorflow as tf
import numpy as np

model = None
output_class = ["Cloudy", "Rain", "Shine", "Sunrise"]
data = {
"Cloudy":
	["Cloudy",
	"4XOAGNzWvqY", "oKFOqMZmuA8"],
"Rain":
	["Rain",
	"Bhi7S06pwv4", "IHPBJySIXZw"],
"Shine":
	["Shine",
	"aUwFXDLOFO0","w0ikFMTuS9c"],
"Sunrise":
	["Sunrise",
	"bYVih298o1Y", "6R8YObQbE88"]
}


def load_artifacts():
    global model
    model = tf.keras.models.load_model("final_model.h5")

def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value, data[predicted_value][0], data[predicted_value][1], data[predicted_value][2]

