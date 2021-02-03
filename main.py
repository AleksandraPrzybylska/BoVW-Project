import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import cluster
import glob
import imutils


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def data_processing(x: np.ndarray) -> np.ndarray:

    resized_x = []
    for i, img_file in enumerate(x):
        img_file = imutils.resize(img_file, width=min(1000, len(img_file[0])))  # resize images
        resized_x.append(img_file)

    return np.asarray(resized_x)


def project():
    np.random.seed(42)

    first_name = 'Aleksandra'
    last_name = 'Przybylska'

    data_path = Path('train/')  # Path to train images directory
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line

    X, y = load_dataset(data_path)
    X = data_processing(X)

    train_images, test_images, train_labels, test_labels = train_test_split(X, y, stratify=y, train_size=0.8, random_state=42)

    # Create a detector/descriptor here. Eg. cv2.AKAZE_create()
    feature_detector_descriptor = cv2.xfeatures2d.SIFT_create()

    # Train a vocabulary model and save it using pickle.dump function
    if './vocab_model.p' not in glob.glob('./*.p'):  # if there is our vocabulary model
    
        NB_WORDS = 800
        kmeans = cluster.MiniBatchKMeans(n_clusters=NB_WORDS, init_size=3 * NB_WORDS)
        train_descriptors = [descriptor for image in train_images for descriptor in
                            feature_detector_descriptor.detectAndCompute(image, None)[1]]
    
        print('Descriptors', len(train_descriptors))
    
        print("Training Vocabulary model")
        kmeans.fit(train_descriptors)
    
        print("Saving Vocabulary Model")
        file_vocab = 'vocab_model.p'
        pickle.dump(kmeans, open(file_vocab, 'wb'))

    with Path('vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)

    # Train a classifier and save it using pickle.dump function
    X_train = apply_feature_transform(train_images, feature_detector_descriptor, vocab_model)
    y_train = train_labels

    X_test = apply_feature_transform(test_images, feature_detector_descriptor, vocab_model)
    y_test = test_labels

    classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=42, tol=1e-5))
    classifier.fit(X_train, y_train)
    file_clf = 'clf.p'
    pickle.dump(classifier, open(file_clf, 'wb'))


    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(X_train, y_train)

    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    # print(f'{first_name} {last_name} score: {score}')
    # with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:  # Don't change the path here
    #     json.dump({'score': score}, score_file)


if __name__ == '__main__':
    project()
