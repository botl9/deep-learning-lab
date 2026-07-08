"""Build viewer/data.js from experiment .py files and metadata.

Usage: python3 viewer/build.py

Generates viewer/data.js with all experiment data and Python code embedded
so the viewer works standalone via file:// or deployed on any static host.
"""

import json, os, re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS = [
    {
        "id": "exp01", "num": "01", "title": "TensorFlow Basics",
        "dataset": "N/A",
        "aim": "To create tensors of various dimensions (scalar, vector, matrix, and higher-order) and perform basic tensor operations such as addition, subtraction, multiplication, division, matrix multiplication, reshaping, and transposition using TensorFlow.",
        "algorithm": [
            "Import TensorFlow library.",
            "Create tensors of different ranks: scalar (0-D), vector (1-D), matrix (2-D), and higher-order (3-D).",
            "Perform element-wise arithmetic operations: addition, subtraction, multiplication, division.",
            "Perform matrix multiplication using tf.matmul.",
            "Reshape and transpose tensors using tf.reshape and tf.transpose.",
            "Print the resulting tensors and verify their shapes.",
        ],
        "parts": [
            {"id": "a", "name": "Creating Tensors",
             "file": "experiment-01/exp01a_tensor_creation.py",
             "image": "exp01a_output.png",
             "result": "Tensors of ranks 0, 1, 2, and 3 were successfully created and their values along with shapes were verified."},
            {"id": "b", "name": "Element-wise Operations",
             "file": "experiment-01/exp01b_elementwise_ops.py",
             "image": "exp01b_output.png",
             "result": "Element-wise arithmetic operations were performed successfully on two 2x2 matrices, confirming that TensorFlow applies operations element-by-element."},
            {"id": "c", "name": "Matrix Multiplication",
             "file": "experiment-01/exp01c_matrix_mul.py",
             "image": "exp01c_output.png",
             "result": "Matrix multiplication using tf.matmul produced the correct dot product of the two matrices."},
            {"id": "d", "name": "Reshape and Transpose",
             "file": "experiment-01/exp01d_reshape_transpose.py",
             "image": "exp01d_output.png",
             "result": "The vector was reshaped from shape (5,) to (5, 1) and the matrix was transposed from (2, 3) to (3, 2) successfully."},
        ],
    },
    {
        "id": "exp02", "num": "02", "title": "Perceptron — Iris",
        "dataset": "Iris",
        "aim": "To design and implement a single unit perceptron from scratch using TensorFlow for binary classification of the Iris dataset without using predefined models.",
        "algorithm": [
            "Load the Iris dataset and extract features and labels.",
            "Convert the multi-class labels to binary (Setosa vs non-Setosa).",
            "Split the dataset into training and testing sets.",
            "Standardize the features using StandardScaler.",
            "Initialize weight vector and bias as TensorFlow Variables.",
            "Define the perceptron forward pass: weighted sum followed by sigmoid activation.",
            "Train using binary cross-entropy loss and gradient descent for 50 epochs.",
            "Compute accuracy on training and testing sets.",
            "Make predictions on test samples and display results.",
        ],
        "parts": [
            {"id": "", "name": "Perceptron from Scratch",
             "file": "experiment-02/exp02_perceptron_iris.py",
             "image": "exp02_output.png",
             "result": "A single unit perceptron was successfully implemented from scratch using TensorFlow. The model classified Iris Setosa vs non-Setosa with high accuracy after 50 epochs of training using binary cross-entropy loss and SGD optimizer."},
        ],
    },
    {
        "id": "exp03", "num": "03", "title": "MLP — Activations & Optimizers",
        "dataset": "Breast Cancer Wisconsin",
        "aim": "To design, train and test a Multi-Layer Perceptron (MLP) on tabular data and compare the performance of different activation functions and optimizers using TensorFlow/Keras.",
        "algorithm": [
            "Load the Breast Cancer Wisconsin dataset from sklearn.datasets.",
            "One-hot encode the labels (2 classes: malignant / benign).",
            "Split data into training and testing sets.",
            "Normalize the features using StandardScaler.",
            "Build an MLP with two hidden layers (8 neurons each) using TensorFlow/Keras Sequential API.",
            "Train the MLP with ReLU, tanh, and sigmoid activation functions using Adam optimizer.",
            "Train the MLP with Adam, SGD, and RMSprop optimizers using ReLU activation.",
            "Evaluate test accuracy for each configuration and compare results.",
        ],
        "parts": [
            {"id": "", "name": "MLP Comparison",
             "file": "experiment-03/exp03_mlp_activations_optimizers.py",
             "image": "exp03_output.png",
             "result": "An MLP with two hidden layers was successfully trained on the Breast Cancer Wisconsin dataset. The performance of different activation functions (ReLU, tanh, sigmoid) and optimizers (Adam, SGD, RMSprop) was compared, demonstrating consistent accuracy above 95% for this binary classification task."},
        ],
    },
    {
        "id": "exp04", "num": "04", "title": "MLP — Fashion MNIST",
        "dataset": "Fashion-MNIST",
        "aim": "To design and implement an MLP to classify 28x28 grayscale fashion images using TensorFlow/Keras and evaluate classification accuracy.",
        "algorithm": [
            "Load the Fashion-MNIST dataset (60,000 training and 10,000 test 28x28 grayscale images, 10 classes).",
            "Combine the training and test sets into a single dataset.",
            "Split the data into training (80%) and testing (20%) sets.",
            "Normalize pixel values to [0, 1] and flatten each image into a 784-dimensional vector.",
            "One-hot encode the labels (10 classes).",
            "Build an MLP with two hidden layers (512 and 256 neurons, ReLU) and a softmax output layer.",
            "Train the model using Adam optimizer and categorical cross-entropy loss for 20 epochs.",
            "Evaluate the model on the test set and report accuracy.",
        ],
        "parts": [
            {"id": "", "name": "MLP Classifier",
             "file": "experiment-04/exp04_mlp_32x32.py",
             "image": "exp04_output.png",
             "result": "An MLP with two hidden layers was trained on the Fashion-MNIST dataset. The model achieved strong classification accuracy on 28x28 grayscale fashion images across 10 categories."},
        ],
    },
    {
        "id": "exp05", "num": "05", "title": "CNN — TF Flowers",
        "dataset": "TF Flowers",
        "aim": "To design and implement a CNN model to classify multi-category JPG images using TensorFlow/Keras and evaluate accuracy with prediction on new images.",
        "algorithm": [
            "Load the TF Flowers dataset using TensorFlow Datasets (5 categories: daisy, dandelion, roses, sunflowers, tulips).",
            "Resize images to 128x128, normalize to [0, 1], convert to NumPy arrays, and one-hot encode labels.",
            "Split the data into training (80%) and testing (20%) sets.",
            "Build a CNN with three convolutional layers (32, 64, 128 filters) each followed by max-pooling, then a softmax output with 5 units.",
            "Train the model for 10 epochs with a batch size of 32.",
            "Evaluate accuracy on the test set.",
            "Predict the label for a new image and display the predicted flower category.",
        ],
        "parts": [
            {"id": "", "name": "CNN Classifier",
             "file": "experiment-05/exp05_cnn_tf_flowers.py",
             "image": "exp05_output.png",
             "result": "A CNN with three convolutional layers was successfully trained on the TF Flowers dataset. The model classified 5 flower categories (daisy, dandelion, roses, sunflowers, tulips) with good accuracy and correctly predicted labels for new images."},
        ],
    },
    {
        "id": "exp06", "num": "06", "title": "CNN — Overfitting & Underfitting",
        "dataset": "MNIST",
        "aim": "To design and implement CNN models on the MNIST dataset to demonstrate overfitting, underfitting, and how regularization techniques (BatchNormalization, Dropout, data augmentation) fix overfitting.",
        "algorithm": [
            "Load the MNIST dataset (10 digit classes, 28x28 grayscale images).",
            "Reshape images to 28x28x1, normalize to [0, 1], and one-hot encode labels.",
            "6a (Overfit): Deep CNN with 3 conv layers (128, 256, 512 filters) and dense layer with no regularization.",
            "6b (Underfit): Shallow CNN with a single 8-filter conv layer and global average pooling.",
            "6c (Fixed): Deep CNN as 6a but with BatchNormalization, Dropout, and data augmentation.",
            "Compare the three approaches to understand overfitting, underfitting, and regularization.",
        ],
        "parts": [
            {"id": "a", "name": "Overfitting",
             "file": "experiment-06/exp06a_overfit.py",
             "image": "exp06a_output.png",
             "result": "The deep CNN without any regularization achieved high training accuracy but significantly lower validation accuracy, confirming overfitting. The model memorized the training data rather than learning generalizable features."},
            {"id": "b", "name": "Underfitting",
             "file": "experiment-06/exp06b_underfit.py",
             "image": "exp06b_output.png",
             "result": "The shallow CNN with only 8 filters and a single convolutional layer performed poorly on both training and validation sets, confirming underfitting. The model lacks the capacity needed to learn meaningful features from the digit images."},
            {"id": "c", "name": "Regularized (Fixed)",
             "file": "experiment-06/exp06c_fixed.py",
             "image": "exp06c_output.png",
             "result": "The fixed CNN with BatchNormalization, Dropout, and data augmentation achieved balanced training and validation performance. The regularization techniques effectively prevented overfitting while the increased capacity avoided underfitting."},
        ],
    },
    {
        "id": "exp07", "num": "07", "title": "CNN — Architectures (LeNet / AlexNet / VGG)",
        "dataset": "MNIST",
        "aim": "To implement and compare classic CNN architectures — LeNet-5, AlexNet, and VGG — for multi-class image classification on the MNIST dataset using TensorFlow/Keras.",
        "algorithm": [
            "Load the MNIST dataset and preprocess (reshape, normalize, one-hot encode).",
            "7a (LeNet-5): Two 5x5 conv layers (6, 16 filters, tanh), average pooling, three dense layers (120, 84, 10). Train 5 epochs.",
            "7b (AlexNet): Five 3x3 conv layers (32, 64, 128, 128, 64 filters, ReLU), max pooling, dense layers with dropout. Train 5 epochs.",
            "7c (VGG): Six 3x3 conv layers (64, 64, 128, 128, 256, 256 filters, ReLU), max pooling after every two convs, dense layers with dropout. Train 5 epochs.",
            "Use Adam optimizer and categorical cross-entropy loss for all three.",
            "Compare parameter counts and test accuracies across the architectures.",
        ],
        "parts": [
            {"id": "a", "name": "LeNet-5",
             "file": "experiment-07/exp07a_lenet.py",
             "image": "exp07a_output.png",
             "result": "LeNet-5, the smallest architecture with approximately 60K parameters, achieved reasonable accuracy on MNIST using tanh activations and average pooling. Its simple structure makes it fast to train but limits capacity compared to deeper networks."},
            {"id": "b", "name": "AlexNet",
             "file": "experiment-07/exp07b_alexnet.py",
             "image": "exp07b_output.png",
             "result": "AlexNet, with its deeper structure (five convolutional layers), ReLU activations, max pooling, and dropout regularization, achieved higher accuracy than LeNet-5. The increased depth and parameter count allowed it to learn more complex features."},
            {"id": "c", "name": "VGG",
             "file": "experiment-07/exp07c_vgg.py",
             "image": "exp07c_output.png",
             "result": "VGG, with six convolutional layers stacked in pairs with max pooling, achieved the highest accuracy due to its greater depth. The use of small 3x3 filters throughout makes it parameter-efficient while allowing deeper feature extraction."},
        ],
    },
    {
        "id": "exp08", "num": "08", "title": "Simple RNN — IMDB",
        "dataset": "IMDB",
        "aim": "To design and implement a Simple RNN model for binary sentiment classification on the IMDB movie review dataset using TensorFlow/Keras.",
        "algorithm": [
            "Load the IMDB dataset from tf.keras.datasets.imdb with the top 10,000 most frequent words.",
            "Pad each review to a fixed length of 500 tokens using pad_sequences.",
            "Build a Sequential model with Embedding (10000 vocab, 128 dim), SimpleRNN (128 units), and Dense (1 unit, sigmoid).",
            "Compile using Adam optimizer and binary cross-entropy loss.",
            "Train for 5 epochs with batch size 64 and 20% validation split.",
            "Evaluate on the test set and print test accuracy.",
        ],
        "parts": [
            {"id": "", "name": "Simple RNN",
             "file": "experiment-08/exp08_simple_rnn.py",
             "image": "exp08_output.png",
             "result": "A Simple RNN model was implemented for binary sentiment classification on the IMDB dataset. The model used an embedding layer followed by a SimpleRNN layer and achieved a reasonable test accuracy, demonstrating the effectiveness of recurrent neural networks for sequence-based text classification tasks."},
        ],
    },
    {
        "id": "exp09", "num": "09", "title": "LSTM — IMDB",
        "dataset": "IMDB",
        "aim": "To design and implement an LSTM model for binary sentiment classification on the IMDB movie review dataset using TensorFlow/Keras.",
        "algorithm": [
            "Load the IMDB dataset from tf.keras.datasets.imdb with the top 10,000 most frequent words.",
            "Pad each review to 500 tokens using pad_sequences.",
            "Build a Sequential model with Embedding (10000 vocab, 128 dim), LSTM (128 units), and Dense (1 unit, sigmoid).",
            "Compile using Adam optimizer and binary cross-entropy loss.",
            "Train for 5 epochs with batch size 64 and 20% validation split.",
            "Evaluate on the test set and print test accuracy.",
        ],
        "parts": [
            {"id": "", "name": "LSTM",
             "file": "experiment-09/exp09_lstm.py",
             "image": "exp09_output.png",
             "result": "An LSTM model was implemented for binary sentiment classification on the IMDB dataset. The model used an embedding layer followed by an LSTM layer and achieved a strong test accuracy, demonstrating the effectiveness of LSTM networks for capturing sequential dependencies in text classification tasks."},
        ],
    },
    {
        "id": "exp10", "num": "10", "title": "GRU — IMDB",
        "dataset": "IMDB",
        "aim": "To design and implement a GRU model for binary sentiment classification on the IMDB movie review dataset using TensorFlow/Keras.",
        "algorithm": [
            "Load the IMDB dataset from tf.keras.datasets.imdb with the top 10,000 most frequent words.",
            "Pad each review to 500 tokens using pad_sequences.",
            "Build a Sequential model with Embedding (10000 vocab, 128 dim), GRU (128 units), and Dense (1 unit, sigmoid).",
            "Compile using Adam optimizer and binary cross-entropy loss.",
            "Train for 5 epochs with batch size 64 and 20% validation split.",
            "Evaluate on the test set and print test accuracy.",
        ],
        "parts": [
            {"id": "", "name": "GRU",
             "file": "experiment-10/exp10_gru.py",
             "image": "exp10_output.png",
             "result": "A GRU model was implemented for binary sentiment classification on the IMDB dataset. The model used an embedding layer followed by a GRU layer and achieved a strong test accuracy, demonstrating the effectiveness of gated recurrent units for sequence-based text classification tasks."},
        ],
    },
    {
        "id": "exp11", "num": "11", "title": "Denoising Autoencoder",
        "dataset": "MNIST",
        "aim": "To design and implement a denoising autoencoder using convolutional layers on the MNIST dataset to remove Gaussian noise from handwritten digit images.",
        "algorithm": [
            "Load the MNIST dataset and normalize pixel values to [0, 1].",
            "Expand dimensions to add a channel axis (28x28x1).",
            "Add Gaussian noise (factor 0.5) to training and test images, then clip to [0, 1].",
            "Build encoder with Conv2D (32, 64 filters) + MaxPooling + bottleneck Conv2D (64).",
            "Build decoder with UpSampling + Conv2D (64, 32 filters) + final Conv2D (1, sigmoid).",
            "Compile using Adam optimizer and binary cross-entropy loss.",
            "Train for 10 epochs with batch size 128, noisy inputs and clean targets.",
            "Evaluate reconstruction loss on noisy test set and display sample statistics.",
        ],
        "parts": [
            {"id": "", "name": "Convolutional Autoencoder",
             "file": "experiment-11/exp11_autoencoder.py",
             "images": ["exp11_output.png", "exp11_output2.png"],
             "result": "A convolutional denoising autoencoder was successfully implemented on the MNIST dataset. The model learned to remove Gaussian noise from handwritten digit images by compressing the input through a bottleneck and reconstructing a clean output, demonstrating the effectiveness of autoencoders for image denoising tasks."},
        ],
    },
    {
        "id": "exp12", "num": "12", "title": "GAN — MNIST",
        "dataset": "MNIST",
        "aim": "To design and implement a Generative Adversarial Network (GAN) to generate handwritten digit images on the MNIST dataset using TensorFlow/Keras.",
        "algorithm": [
            "Load the MNIST dataset and normalize pixel values to [0, 1].",
            "Build the Generator: Dense layers (128, 256, 512, 784 units with ReLU, final Sigmoid) reshaping to 28x28x1.",
            "Build the Discriminator: Flatten, Dense (512, 256 ReLU), Sigmoid output for real/fake classification.",
            "Compile Discriminator using Adam (lr=0.0002) and binary cross-entropy loss; freeze it, then create Combined model (Generator -> Discriminator).",
            "For each epoch: sample random noise, generate fake images, train Discriminator on real (label=1) and fake (label=0) batches.",
            "Train Generator via Combined model with noise input and valid labels (label=1) to fool the Discriminator.",
            "Print Discriminator loss/accuracy and Generator loss for each epoch.",
            "After training, generate 10 sample digits and display pixel mean statistics.",
        ],
        "parts": [
            {"id": "", "name": "GAN",
             "file": "experiment-12/exp12_gan.py",
             "image": "exp12_output.png",
             "result": "A Generative Adversarial Network was implemented on the MNIST dataset. The Generator learned to produce realistic handwritten digit images from random noise, while the Discriminator simultaneously improved at distinguishing real from fake samples. The adversarial training process successfully balanced both networks, demonstrating the effectiveness of GANs for generative image tasks."},
        ],
    },
]


def js_str(s):
    """Escape a Python string for use as a JS template literal."""
    s = s.replace("\\", "\\\\")
    s = s.replace("`", "\\`")
    s = s.replace("${", "\\${")
    return s


def read_code(filepath):
    full = os.path.join(REPO_ROOT, filepath)
    with open(full, "r") as f:
        return f.read().rstrip("\n")


def build():
    parts = []
    parts.append("// Auto-generated by build.py — do not edit directly")
    parts.append("// Re-generate with: python3 viewer/build.py")
    parts.append("")
    parts.append("const experiments = [")

    for exp in EXPERIMENTS:
        code_indent = "        "
        parts.append("  {")
        for key in ("id", "num", "title", "dataset", "aim"):
            val = js_str(exp[key])
            parts.append(f'    {key}: `{val}`,')
        parts.append("    algorithm: [")
        for step in exp["algorithm"]:
            parts.append(f'      `{js_str(step)}`,')
        parts.append("    ],")
        parts.append("    parts: [")
        for part in exp["parts"]:
            code = read_code(part["file"])
            parts.append("      {")
            for key in ("id", "name"):
                val = js_str(part[key])
                parts.append(f'        {key}: `{val}`,')
            parts.append(f'        file: `{part["file"]}`,')
            if "images" in part:
                imgs = json.dumps(part["images"])
                parts.append(f"        images: {imgs},")
            elif "image" in part:
                parts.append(f'        image: `{part["image"]}`,')
            parts.append(f"        code: `{js_str(code)}`,")
            parts.append(f'        result: `{js_str(part["result"])}`,')
            parts.append("      },")
        parts.append("    ],")
        parts.append("  },")

    parts.append("];")
    parts.append("")

    out = "\n".join(parts)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.js")
    with open(out_path, "w") as f:
        f.write(out)

    print(f"Wrote {out_path} ({len(out)} bytes)")


if __name__ == "__main__":
    build()
