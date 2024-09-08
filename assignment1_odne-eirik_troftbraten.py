# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

songs = pd.read_csv("SpotifyFeatures.csv", delimiter = ",")

songs_rows, songs_columns = songs.shape

# Splits the 2 genres in to new different datasamples, prints the amount of each genre and then merge them
pop_songs = songs[songs["genre"]=="Pop"]
classical_songs = songs[songs["genre"]=="Classical"]
pop_songs_amount = len(pop_songs)
classical_songs_amount = len(classical_songs)
print(f"Amount of pop songs: {pop_songs_amount}\nAmount of classical songs: {classical_songs_amount}")
selected_songs = pd.concat([pop_songs,classical_songs])

selected_songs["Label"] = selected_songs["genre"].apply(lambda x: 1 if x == "Pop" else 0)   # Adding the labels 1 and 0 to the samples according to the genre

feature_matrix = np.array(selected_songs.loc[:,"liveness":"loudness"])  # Making a matrix with the pop and classical samples with only liveness and loudness as features
X = feature_matrix

# Finds the smalles and highest value in the samples features
min_feature = np.min(X)
max_feature = np.max(X)
print(f"Smallest value: {min_feature}\nHighest value: {max_feature}")

label_vector = np.array(selected_songs["Label"])
y = label_vector

x_train, x_test, y_train, y_test = train_test_split(feature_matrix, label_vector, test_size = 0.2, random_state = 42)   # Splitting up the matrix and y-vector into a training set and a testing set with sklearn import

# Sigmoid function for logistic regression
def sigmoid(z):
    z = np.clip(z, -700, 700)   # Got overflow error, found this solution from a youtube-comment. Limits the values of z
    return 1 / (1 + np.exp(-z))

def loss_func(y, y_hat):
    epsilon = 1e-7  # Adding a small value to avoid ValueError with log(0)
    n = y.shape[0]
    loss = -np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
    cost = loss / n
    return cost

def train(X, y, lr, epochs, batch_size=364):
    # Initializing weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0.0

    costs = []  
    n_epochs = []
    
    for epoch in range(epochs): # For each epoch, takes random batches and processes the entire dataset
        slice = np.random.permutation(X.shape[0])   # Makes random indexes to shuffle the dataset
        # Shuffles the set with random indexes
        X_shuffled = X[slice]   
        y_shuffled = y[slice]   
        
        for i in range(0, X.shape[0], batch_size):  # Loops over the shuffled set in batches
            # Takes the i'th batch with features and its labels
            X_batch = X_shuffled[i:(i + batch_size)]
            y_batch = y_shuffled[i:(i + batch_size)]

            z = np.dot(X_batch, weights) + bias # Does linear regression before applying the sigmoid function
            y_hat = sigmoid(z)  # Sends z to the sigmoid function to get predictions between 0 and 1
            
            # Gradient computing of the loss function
            gradient = y_hat - y_batch  
            d_w = np.dot(X_batch.T, gradient) / batch_size  # Gradient with focus on the weight
            d_b = np.sum(gradient) / batch_size # Gradient with focus on the bias
            # Updating the weight and bias after calculating the error in biases and weights, scaled by lr (learning rate)
            weights -= lr * d_w
            bias -= lr * d_b
        
        # Does prediction on the whole dataset with updated weights and biases
        lin_test = np.dot(X, weights) + bias
        sig_test = sigmoid(lin_test)
        loss = loss_func(y, sig_test) # Calculates the cost for this epoch
        costs.append(loss)
        n_epochs.append(epoch)
        
        #Plots and shows the cost as a function of epochs
        plt.plot(n_epochs, costs)
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")

        # Shows and update every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, cost = {loss}")
    
    return weights, bias, costs

weights, bias, cost = train(x_train, y_train, lr=0.01, epochs=100)

# Predict function using the finished updated and trained weights and bias
def predict(X,weights,bias):
    z = np.dot(X,weights) + bias
    y_pred = sigmoid(z)
    return np.where((y_pred >=0.5), 1, 0)

# Calculates the accuracy of the model
def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    accuracy = correct/len(y_true)
    return accuracy

# Predicts using the already trained dataset
y_pred_train = predict(x_train, weights, bias)
train_accuracy = accuracy(y_train, y_pred_train)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Predicts using the leftover testin dataset
y_pred_test = predict(x_test, weights, bias)
test_accuracy = accuracy(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Checks if the model is usable on new, unseen data
if abs(train_accuracy - test_accuracy) > 0.05:
    print("There is  a significant difference between the finished training set and the new testing set")
else:
    print("The results from the finished training set and the new testing set are similar")


# Makes a confussion matrix to check the results
confusion_matrix = metrics.confusion_matrix(y_test,y_pred_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0,1])
cm_display.plot()
plt.show()