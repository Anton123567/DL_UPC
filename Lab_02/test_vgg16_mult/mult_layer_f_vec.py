import numpy as np
import tensorflow as tf
import cv2


def full_network_embedding(model, image_paths, batch_size, target_layer_names, input_reshape, stats=None):
    '''
    Generates the Full-Network embedding[1] of a list of images using a pre-trained
    model (input parameter model) with its computational graph loaded. Tensors used
    to compose the FNE are defined by target_tensors input parameter. The input_tensor
    input parameter defines where the input is fed to the model.

    By default, the statistics used to standardize are the ones provided by the same
    dataset we wish to compute the FNE for. Alternatively these can be passed through
    the stats input parameter.

    This function aims to generate the Full-Network embedding in an illustrative way.
    We are aware that it is possible to integrate everything in a tensorflow operation,
    however this is not our current goal.

    [1] https://arxiv.org/abs/1705.07706

    Args:
        model (tf.GraphDef): Serialized TensorFlow protocol buffer (GraphDef) containing the pre-trained model graph
                             from where to extract the FNE. You can get corresponding tf.GraphDef from default Graph
                             using `tf.Graph.as_graph_def`.
        image_paths (list(str)): List of images to generate the FNE for.
        batch_size (int): Number of images to be concurrently computed on the same batch.
        target_layer_names (list(str)): List of tensor names from model to extract features from.
        input_reshape (tuple): A tuple containing the desired shape (height, width) used to resize the image.
        stats (2D ndarray): Array of feature-wise means and stddevs for standardization.

    Returns:
       2D ndarray : List of features per image. Of shape <num_imgs,num_feats>
       2D ndarry: Mean and stddev per feature. Of shape <2,num_feats>
    '''

    # Define feature extractor
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in model.layers if layer.name in target_layer_names],
    )
    get_raw_features = lambda x: [tensor.numpy() for tensor in feature_extractor(x)]

    # Prepare output variable
    feature_shapes = [layer.output_shape for layer in model.layers if layer.name in target_layer_names]
    len_features = sum(shape[-1] for shape in feature_shapes)
    features = np.empty((len(image_paths), len_features))

    # Extract features
    for idx in range(0, len(image_paths), batch_size):
        batch_images_path = image_paths[idx:idx + batch_size]
        img_batch = np.zeros((len(batch_images_path), *input_reshape, 3), dtype=np.float32)
        for i, img_path in enumerate(batch_images_path):
            cv_img = cv2.imread(img_path)
            try:
                cv_img_resize = cv2.resize(cv_img, input_reshape)
                img_batch[i] = np.asarray(cv_img_resize, dtype=np.float32)[:, :, ::-1]
            except:
                print(img_path)

        feature_vals = get_raw_features(img_batch)
        features_current = np.empty((len(batch_images_path), 0))
        for feat in feature_vals:
            # If its not a conv layer, add without pooling
            if len(feat.shape) != 4:
                features_current = np.concatenate((features_current, feat), axis=1)
                continue
            # If its a conv layer, do SPATIAL AVERAGE POOLING
            pooled_vals = np.mean(np.mean(feat, axis=2), axis=1)
            features_current = np.concatenate((features_current, pooled_vals), axis=1)
        # Store in position
        features[idx:idx + len(batch_images_path)] = features_current.copy()

    # STANDARDIZATION STEP
    # Compute statistics if needed
    if stats is None:
        stats = np.zeros((2, len_features))
        stats[0, :] = np.mean(features, axis=0)
        stats[1, :] = np.std(features, axis=0)
    # Apply statistics, avoiding nans after division by zero
    features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features), where=stats[1] != 0)
    if len(np.argwhere(np.isnan(features))) != 0:
        raise Exception('There are nan values after standardization!')
    # DISCRETIZATION STEP
    th_pos = 0.15
    th_neg = -0.25
    features[features > th_pos] = 1
    features[features < th_neg] = -1
    features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

    # # Store output
    # outputs_path = '../outputs'
    # if not os.path.exists(outputs_path):
    #     os.makedirs(outputs_path)
    # np.save(os.path.join(outputs_path, 'fne.npy'), features)
    # np.save(os.path.join(outputs_path, 'stats.npy'), stats)

    # Return
    return features, stats


import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    # This shows an example of calling the full_network_embedding method using
    # the VGG16 architecture pretrained on ILSVRC2012 (aka ImageNet), as
    # provided by the keras package. Using any other pretrained CNN
    # model is straightforward.

    # Load model
    img_width, img_height = 256, 256
    model = load_model('model.h5')
    initial_model = load_model('model.h5')

    target_layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1',
                          'block3_conv2',
                          'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1',
                          'block5_conv2',
                          'block5_conv3', 'fc1', 'fc2']

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

    train_images = []
    train_labels = []
    # Use a subset of classes to speed up the process. -1 uses all classes.

    for train_image, label in zip(train_df["file_path"], train_df["label"]):
        train_images.append(train_image)
        train_labels.append(label)

    val_images = []
    val_labels = []

    for val_image, label in zip(val_df["file_path"], val_df["label"]):
        val_images.append(val_image)
        val_labels.append(label)

    print('Total train images:', len(train_images), ' with their corresponding', len(train_labels), 'labels')
    # Parameters for the extraction procedure
    batch_size = 1
    input_reshape = (256, 256)
    # Call FNE method on the train set
    fne_features, fne_stats_train = full_network_embedding(initial_model, train_images, batch_size,
                                                           target_layer_names, input_reshape)
    print('Done extracting features of training set. Embedding size:', fne_features.shape)

    from sklearn import svm

    # Train SVM with the obtained features.
    clf = svm.LinearSVC()
    clf.fit(X=fne_features, y=train_labels)
    print('Done training SVM on extracted features of training set')

    fne_features, fne_stats_train = full_network_embedding(initial_model, val_images, batch_size,
                                                           target_layer_names, input_reshape, stats=fne_stats_train)
    print('Done extracting features of val set')

    # Test SVM with the test set.
    predicted_labels = clf.predict(fne_features)
    print('Done testing SVM on extracted features of test set')


    # Print results
    report = classification_report(val_labels, predicted_labels, output_dict=True)#
    df = pd.DataFrame(report).transpose()
    df.to_csv("report_svm")

    print(classification_report(val_labels, predicted_labels))
    cm = confusion_matrix(val_labels, predicted_labels)
    print(cm)


    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Saving the plot as a PNG file
    plt.savefig('confusion_matrix.png')



