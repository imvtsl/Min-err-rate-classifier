import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

ORANGE = 'orange'

GREEN = 'green'


def get_key_name(file_name):
    key = None
    if file_name == 'training_data_5.mat':
        key = 'train_data_5'
    if file_name == 'training_data_6.mat':
        key = 'train_data_6'
    if file_name == 'testing_data_5.mat':
        key = 'test_data_5'
    if file_name == 'testing_data_6.mat':
        key = 'test_data_6'
    return key


def get_sample_size(file_name):
    size = None
    if file_name == 'training_data_5.mat':
        size = 5421
    if file_name == 'training_data_6.mat':
        size = 5918
    if file_name == 'testing_data_5.mat':
        size = 892
    if file_name == 'testing_data_6.mat':
        size = 958
    return size


def vectorize(file_name):
    mat_data = scipy.io.loadmat(file_name)
    print(type(mat_data))
    # type is dictionary
    # print(mat_data)

    for key in mat_data:
        print(key)

    # verified manually that variable name is train_data_5 and so on
    key = get_key_name(file_name)

    image_data = mat_data[key]

    print(type(image_data))
    # type is ndarray

    # check shape
    print(image_data.shape)

    # shape is tuple of size 3.
    # verify sizes and shape
    size = get_sample_size(file_name)
    image_data_status = verify_image_data_shape(size, image_data)
    if not image_data_status:
        print('data in {0} is not correct'.format(file_name))
        return
    else:
        print('data in {0} is verified'.format(file_name))

    # vectorize image
    vectorized_images = image_data.reshape(image_data.shape[0], -1)
    print('data in {0} is vectorized'.format(file_name))

    # verifying size and shape
    vectorized_status = verify_vectorized_shape(size, vectorized_images)
    if not vectorized_status:
        print('data in {0} is not correctly vectorized'.format(file_name))
        return
    else:
        print('verified that data in {0} is correctly vectorized'.format(file_name))

    return vectorized_images


def verify_image_data_shape(size, image_data):
    if image_data.shape[0] != size or image_data.shape[1] != 28 or image_data.shape[2] != 28:
        return False
    return True


def verify_vectorized_shape(size, vectorized_images):
    if vectorized_images.shape[0] != size or vectorized_images.shape[1] != 784:
        return False
    return True


def verify_normalized_shape(size, normalized_data):
    if normalized_data.shape[0] != size or normalized_data.shape[1] != 784:
        return False
    return True


def project():
    training_data_5_file_name = 'training_data_5.mat'
    testing_data_5_file_name = 'testing_data_5.mat'

    training_data_6_file_name = 'training_data_6.mat'
    testing_data_6_file_name = 'testing_data_6.mat'

    # vectorize data
    training_data_5_vectorized = vectorize(training_data_5_file_name)
    testing_data_5_vectorized = vectorize(testing_data_5_file_name)

    training_data_6_vectorized = vectorize(training_data_6_file_name)
    testing_data_6_vectorized = vectorize(testing_data_6_file_name)

    # combine training set, then normalise
    training_data_combined = np.concatenate((training_data_5_vectorized, training_data_6_vectorized), axis=0)
    testing_data_combined = np.concatenate((testing_data_5_vectorized, testing_data_6_vectorized), axis=0)

    # Task 1. Feature normalization(Data conditioning)
    training_data_normalized, testing_data_normalized = normalize(training_data_combined, testing_data_combined)

    # Task 2. PCA using the training samples
    cov_matrix = np.cov(training_data_normalized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the Eigenvalues and Eigenvectors in Descending Order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    number_of_components = 2
    principal_components = eigenvectors[:, :number_of_components]

    # Task 3. Dimension reduction using PCA
    training_proj, testing_proj = dimensionality_reduction(principal_components,
                                                           training_data_normalized,
                                                           testing_data_normalized,
                                                           )

    # Task 4. Density estimation
    multivariate_distribution_5, multivariate_distribution_6, mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6 = get_multivariate_distributions(training_proj)

    # Task 5. Bayesian Decision Theory for optimal classification
    training_accuracy, testing_accuracy = minimum_error_rate_classification(mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6, training_proj,
                                                                      testing_proj)
    print(f"Accuracy on the training set: {training_accuracy * 100:.2f}%")
    print(f"Accuracy on the testing set: {testing_accuracy * 100:.2f}%")



def data_classification(data, mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6):
    # Calculate probabilities for each digit
    pdf_class_5 = multivariate_normal.pdf(data, mean=mean_class_5, cov=covariance_matrix_class_5)
    pdf_class_6 = multivariate_normal.pdf(data, mean=mean_class_6, cov=covariance_matrix_class_6)

    print("pdf digit 5")
    print(pdf_class_5)
    print("pdf digit 6")
    print(pdf_class_6)

    # Assign data to the digit with the higher probability
    if pdf_class_5 > pdf_class_6:
        # Digit 5
        return 0
    else:
        # Digit 6
        return 1

def minimum_error_rate_classification(mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6, training_proj,
                                      testing_proj):

    # priors are equal, so only likelihood makes the difference.

    # Adding labels to training and testing data
    training_proj = np.hstack((training_proj, np.zeros((11339, 1))))
    training_proj[5421:, 2] = 1

    testing_proj = np.hstack((testing_proj, np.zeros((1850, 1))))
    testing_proj[892:, 2] = 1

    # Initialize variables to keep track of correct classifications
    number_correct_training = 0
    number_correct_testing = 0

    for data in training_proj:
        # Last column is the class label
        actual_class = data[-1]
        predicted_class = data_classification(data[:-1], mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6)
        print(predicted_class)
        if actual_class == predicted_class:
            number_correct_training += 1

    for data in testing_proj:
        # Last column is the class label
        actual_class = data[-1]

        # Excluding the class label while data classification
        predicted_class = data_classification(data[:-1], mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6)
        print(predicted_class)

        if actual_class == predicted_class:
            number_correct_testing += 1

    # Calculate accuracy for both training and testing sets
    number_training_samples_total = len(training_proj)
    number_testing_samples_total = len(testing_proj)
    print(number_training_samples_total)
    print(number_testing_samples_total)
    print(number_correct_training)
    print(number_correct_testing)

    training_accuracy = number_correct_training / number_training_samples_total
    testing_accuracy = number_correct_testing / number_testing_samples_total
    return training_accuracy, testing_accuracy

def get_multivariate_distributions(training_proj):
    class_5_indices_end = 5421
    # Calculate the mean (mu) for each class
    mean_class_5 = np.mean(training_proj[:class_5_indices_end], axis=0)
    mean_class_6 = np.mean(training_proj[class_5_indices_end:], axis=0)

    print("mean for digit 5:")
    print(mean_class_5)
    print("mean for digit 6:")
    print(mean_class_6)

    # Calculate the covariance matrix (sigma) for each class
    covariance_matrix_class_5 = np.cov(training_proj[:class_5_indices_end], rowvar=False)
    covariance_matrix_class_6 = np.cov(training_proj[class_5_indices_end:], rowvar=False)

    print("covariance matrix for digit 5")
    print(covariance_matrix_class_5)
    print("covariance matrix for digit 6")
    print(covariance_matrix_class_6)


    multivariate_distribution_5 = multivariate_normal(mean_class_5, covariance_matrix_class_5)
    multivariate_distribution_6 = multivariate_normal(mean_class_6, covariance_matrix_class_6)

    return multivariate_distribution_5, multivariate_distribution_6, mean_class_5, mean_class_6, covariance_matrix_class_5, covariance_matrix_class_6


def dimensionality_reduction(principal_components, training_data_normalized,
                             testing_data_normalized):
    training_proj = np.dot(training_data_normalized, principal_components)
    # print(type(training_proj_5))
    testing_proj = np.dot(testing_data_normalized, principal_components)

    class_5_indices_end = 5421

    # Plot the training data for digit 5 in green
    plt.scatter(training_proj[:class_5_indices_end, 0], training_proj[:class_5_indices_end, 1], c=GREEN,
                label='Digit 5 (Training)', alpha=0.5)

    # Plot the training data for digit 6 in orange
    plt.scatter(training_proj[class_5_indices_end:, 0], training_proj[class_5_indices_end:, 1], c=ORANGE,
                label='Digit 6 (Training)', alpha=0.5)

    # Set legend and labels
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.legend()

    # Show the plot
    plt.title('2-D Projection of Training Data for Digit 5 and Digit 6')
    plt.show()

    # plot PCA
    plt.figure(figsize=(12, 6))

    # Plot histograms for Digit 5 and Digit 6 along the first principal component
    plt.hist(training_proj[:class_5_indices_end, 0], bins=30, alpha=0.5, color=GREEN, label='Digit 5 - PC1')
    plt.hist(training_proj[class_5_indices_end:, 0], bins=30, alpha=0.5, color=ORANGE, label='Digit 6 - PC1')

    # Set legend and labels
    plt.xlabel('1st Principal Component')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot
    plt.title('Histograms for the First Principal Component')
    plt.show()

    # Repeating same steps for the second principal component

    # Create histograms for the second principal component
    plt.figure(figsize=(12, 6))

    # Plot histograms for Digit 5 and Digit 6 along the second principal component
    plt.hist(training_proj[:class_5_indices_end, 1], bins=30, alpha=0.5, color=GREEN, label='Digit 5 - PC2')
    plt.hist(training_proj[class_5_indices_end:, 1], bins=30, alpha=0.5, color=ORANGE, label='Digit 6 - PC2')

    # Set legend and labels
    plt.xlabel('2nd Principal Component')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot
    plt.title('Histograms for the Second Principal Component')
    plt.show()

    return training_proj, testing_proj


def normalize(training_data_vectorized, testing_data_vectorized):
    mean = np.mean(training_data_vectorized, axis=0)
    std = np.std(training_data_vectorized, axis=0)

    # Avoiding division by zero by adding a small constant
    epsilon = 1e-8
    std = np.where(std == 0, epsilon, std)

    normalized_training_data = (training_data_vectorized - mean) / std
    normalized_testing_data = (testing_data_vectorized - mean) / std

    # verifying size and shape
    training_data_size = get_sample_size('training_data_5.mat') + get_sample_size('training_data_6.mat')
    normalized_training_data_status = verify_normalized_shape(training_data_size, normalized_training_data)
    if not normalized_training_data_status:
        print('training data is not correctly normalized')
        return
    else:
        print('verified that training data may be correctly normalized')

    testing_data_size = get_sample_size('testing_data_5.mat') + get_sample_size('testing_data_6.mat')
    normalized_testing_data_status = verify_normalized_shape(testing_data_size, normalized_testing_data)
    if not normalized_testing_data_status:
        print('testing data is not correctly normalized')
        return
    else:
        print('verified that testing data may be correctly normalized')

    return normalized_training_data, normalized_testing_data


if __name__ == '__main__':
    project()
