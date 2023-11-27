import pandas as pd



dataset = pd.read_csv("./../../DataMeta/MAMe_dataset.csv")
labels = pd.read_csv("./../../DataMeta/MAMe_labels.csv", header=None)
toy_data = pd.read_csv("./../../DataMeta/MAMe_toy_dataset.csv")

important = dataset[["Image file", "Subset", "Medium"]]
important = important.rename(columns={"Medium": "label"})
important = important.rename(columns={"Image file": "file_path"})
important["file_path"] = important["file_path"].apply(lambda x: "./../../DataProcessed/data_256/" + str(x))

print("Mapping labels...")
label_mapper = labels.to_dict()[1]
label_mapper = {v: k for k, v in label_mapper.items()}
important["label"] = important["label"].map(label_mapper)
important = important.dropna()
important["label"].astype(int)
# number_classes = len(important["label"].drop_duplicates())

# important = pd.get_dummies(important, columns=["label"], prefix="label_")
# labels = [x for x in important.columns if "label" in x]

print("Creating train, val, test dfs...")

train_df = important[important['Subset'] == 'train']
val_df = important[important['Subset'] == 'val']
test_df = important[important['Subset'] == 'test']

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

#%%


import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image

# Load the pretrained VGG16 model
#model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))  # Set include_top=False to use as a feature extractor
model = load_model('model.h5')


#%%
import numpy as np
def load_and_process_image(img_path):
    """ Load and preprocess a single image. """
    img = Image.open(img_path)
    img = img.resize((256, 256))  # VGG16 expects images of size 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def process_images(df, name):
    """ Process all images in a given folder. """
    embedding_list = []
    label_list = []
    feature_df = df.copy()
    feature_df["embedding"] = None

    for idx, img_path in df["file_path"].items():
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_array = load_and_process_image(img_path)
            v = model.predict(img_array)
            feature_df.loc[idx, "embedding"] = v
            #feature_df.to_csv("features_" + name + ".csv")
            embedding_list.append(v)
            label_list.append(df.loc[idx, "label"])

    return np.vstack(embedding_list), np.vstack(label_list)




tr_emb, tr_lab = process_images(train_df, "train")
val_emb, val_lab = process_images(val_df, "val")
test_emb, test_lab = process_images(test_df, "test")

from sklearn import svm

# Train SVM with the obtained features.
clf = svm.LinearSVC()
clf.fit(X=tr_emb, y=tr_lab)
print('Done training SVM on extracted features of training set')

print('Done extracting features of val set')

# Test SVM with the test set.
predicted_labels = clf.predict(val_emb)
print('Done testing SVM on extracted features of test set')
from sklearn.metrics import classification_report, confusion_matrix

# Print results
report = classification_report(val_lab, predicted_labels, output_dict=True)  #
df = pd.DataFrame(report).transpose()
df.to_csv("report_svm")

print(classification_report(val_lab, predicted_labels))
cm = confusion_matrix(val_lab, predicted_labels)
print(cm)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Saving the plot as a PNG file
plt.savefig('confusion_matrix.png')


# 'features' now contains the VGG16 features for the images in the specified folder



