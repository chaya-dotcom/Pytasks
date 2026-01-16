import os
#from keras.layers import Input, Lambda, Dense, Flatten, Dropout
#from keras.models import Model 
#from keras.applications.vgg19 import VGG19
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping
#from keras.models import load_model
import os
#from keras.layers import Input, Lambda, Dense, Flatten, Dropout
#from keras.models import Model 
#from keras.applications.vgg19 import VGG19
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping
#from keras.models import load_model
import os
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model 
from keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import load_model






import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter.filedialog
from tkinter import *
from PIL import Image, ImageTk, ImageOps
import time

IMAGE_SIZE = [224, 224]

train_path = "dataset/train"
test_path = "dataset/test"
val_path = "dataset/val"

# ---------------- Data Loading ----------------
x_train, x_test, x_val = [], [], []

for folder in os.listdir(train_path):
    for img in os.listdir(os.path.join(train_path, folder)):
        img_arr = cv2.imread(os.path.join(train_path, folder, img))
        img_arr = cv2.resize(img_arr, (224, 224))
        x_train.append(img_arr)

for folder in os.listdir(test_path):
    for img in os.listdir(os.path.join(test_path, folder)):
        img_arr = cv2.imread(os.path.join(test_path, folder, img))
        img_arr = cv2.resize(img_arr, (224, 224))
        x_test.append(img_arr)

for folder in os.listdir(val_path):
    for img in os.listdir(os.path.join(val_path, folder)):
        img_arr = cv2.imread(os.path.join(val_path, folder, img))
        img_arr = cv2.resize(img_arr, (224, 224))
        x_val.append(img_arr)

train_x, test_x, val_x = np.array(x_train)/255.0, np.array(x_test)/255.0, np.array(x_val)/255.0

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path, target_size=(224,224), batch_size=32, class_mode='sparse')
test_set = test_datagen.flow_from_directory(test_path, target_size=(224,224), batch_size=32, class_mode='sparse')
val_set = val_datagen.flow_from_directory(val_path, target_size=(224,224), batch_size=32, class_mode='sparse')

train_y, test_y, val_y = training_set.classes, test_set.classes, val_set.classes

# ---------------- Model ----------------
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(3, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

# ---------------- GUI Functions ----------------
def load_dataset():
    status_label.config(text="Loading Dataset...")
    root.update_idletasks()
    time.sleep(2)
    status_label.config(text="Dataset loaded successfully")

def preprocess():
    status_label.config(text="Preprocessing...")
    root.update_idletasks()
    time.sleep(2)
    status_label.config(text="Image resized successfully")

def train():
    status_label.config(text="Training...")
    root.update_idletasks()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

#epochs=10

    history = model.fit(train_x, train_y, validation_data=(val_x,val_y), 
                        epochs=10,
                        callbacks=[early_stop], batch_size=2, shuffle=True)

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('vgg-loss-rps-1.png')

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('vgg-acc-rps-1.png')

    model.save("vgg-rps-final.h5")
    status_label.config(text="Trained Successfully")

def show_loss():
    global panelA
    img = cv2.imread("vgg-loss-rps-1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(img).resize((300, 305), Image.Resampling.LANCZOS)
    im2 = ImageTk.PhotoImage(im2)
    panelA.configure(image=im2) if panelA else None
    panelA.image = im2
    panelA.place(x=430, y=150)
    status_label.config(text="Loss Graph")

def show_acc():
    global panelA
    img = cv2.imread("vgg-acc-rps-1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(img).resize((300, 305), Image.Resampling.LANCZOS)
    im2 = ImageTk.PhotoImage(im2)
    panelA.configure(image=im2) if panelA else None
    panelA.image = im2
    panelA.place(x=430, y=150)
    status_label.config(text="Accuracy Graph")

def detection():
    path = tkinter.filedialog.askopenfilename()
    if len(path) > 0:
        model = load_model('vgg-rps-final.h5')

        # Prepare image for model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        pil_img = Image.open(path).convert("RGB")
        pil_resized = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)
        arr = np.asarray(pil_resized)
        data[0] = (arr.astype(np.float32) / 127.0) - 1

        prediction = model.predict(data)
        y_classes = prediction.argmax(axis=-1)[0]   # take first element
        accu = prediction[0][y_classes] * 100

        # ---- Disease labels + extra info ----
        if y_classes == 0:
            val = "Melanoma"
            info = "⚠️ Dangerous skin cancer\nNeeds urgent diagnosis"
        elif y_classes == 1:
            val = "Normal Skin"
            info = "✅ No disease detected"
        else:
            val = "Keratosis Viridis"
            info = "⚠️ Precancerous lesion\nNeeds monitoring"

        # ---- OpenCV lesion detection ----
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = None

        if val != "Normal Skin":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cropped = img_rgb[y:y+h, x:x+w]

        if cropped is None:
            cropped = np.ones((100, 300, 3), np.uint8)*255
            cv2.putText(cropped, "No lesion detected", (5,50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,0), 2)

        # ---- Convert to Tkinter Images ----
        original_img = ImageTk.PhotoImage(Image.fromarray(img_rgb).resize((300,300)))
        cropped_img = ImageTk.PhotoImage(Image.fromarray(cropped).resize((300,300)))

        global panelA, panelB

        # Left = original
        if panelA is None:
            panelA = Label(root, image=original_img)
            panelA.place(x=350, y=150)
        else:
            panelA.configure(image=original_img)
        panelA.image = original_img

        # Right = cropped
        if panelB is None:
            panelB = Label(root, image=cropped_img)
            panelB.place(x=680, y=150)
        else:
            panelB.configure(image=cropped_img)
        panelB.image = cropped_img

        # ---- Status text with accuracy & disease ----
        status_label.config(
            text=f"Accuracy: {accu:.2f}%\nPredicted: {val}\n{info}"
        )

# ---------------- GUI ----------------
root = Tk()
root.geometry("1000x700")
root.title("Skin Disease Detection")
root.configure(bg="#DAF7A6")

Label(root, text="Skin Disease Detection Using Machine Learning",
      font=("Courier", 18,"italic","bold"), fg="Red").place(x=450, y=25, anchor="center")

Label(root, text=">>> MENU <<<", font=("Courier", 14,"italic","bold"),fg="Black").place(x=60, y=50)

Button(root, text="Load Dataset", command=load_dataset, bg="#f9a655",font=("Courier", 14,"italic","bold"),height=1, width=15).place(x=50, y=100)
Button(root, text="Preprocessing", command=preprocess, bg="#f9a655",font=("Courier", 14,"italic","bold"),height=1, width=15).place(x=50, y=150)
Button(root, text="Train", command=train, bg="#f9a655",font=("Courier", 14,"italic","bold"),height=1, width=15).place(x=50, y=200)
Button(root, text="Loss Graph", command=show_loss, bg="#f9a655",font=("Courier", 14,"italic","bold"),height=1, width=15).place(x=50, y=250)
Button(root, text="Accuracy Graph", command=show_acc, bg="#f9a655",font=("Courier", 14,"italic","bold"),height=1, width=15).place(x=50, y=300)
Button(root, text="Browse Image", command=detection, bg="#f9a655",font=("Courier", 14,"italic","bold"),height=1, width=15).place(x=50, y=350)

status_label = Label(root, font=("Courier", 14,"italic","bold"),fg="Black")
status_label.place(x=400,y=600)

# panels for images
panelA, panelB = None, None

root.mainloop()
