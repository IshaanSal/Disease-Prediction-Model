# Disease-Prediction-Model
Training a neural network to predict a certain disease based on the described symptoms.

This project aims to showcase the extent to which neural networks are effective as an ML algorithm in the context of predicting diseases in accordance with the symptoms.
After discovering a Kaggle dataset which consisted of a set of symptoms, as well as the disease that corresponded with this combination of symptoms, I attempted to develop an Artificial Neural Network that was capable of predicting the disease based on some given set of symptoms. While other classification algorithms such as decision trees or random forest could have been applicable for this situation, I determined that neural networks would provide the best opportunity for effective predictions.

### Neural Networks Explained
A neural network is a type of deep learning algorithm that is modeled after the structure and function of the human brain. It is a collection of interconnected nodes, or neurons, that are organized into layers. Each neuron in the network takes in one or more inputs, applies a mathematical function to those inputs, and then outputs a result. These outputs are then used as inputs for the next layer of neurons in the network, and so on, until a final output is produced.
A neural network typically consists of three types of layers: input layers, hidden layers, and output layers. The input layer is where the initial data is fed into the network. The hidden layers perform mathematical operations on the input data and pass the results to the output layer. The output layer produces the final output, which could be a prediction, a classification, or any other desired output. In the case of this model, the NN(neural network) ultimately predicts what disease is being described from the symptoms.

### Architecture Details
After the data was imported, the first 132 columns indicate the presence of a symptom. Each column represents a different symptom, with a 1 under each column representing the symptom being active in the disease, while a 0 under each column indicates the lack of that respective symptom. Following these columns, the 133rd column is the prognosis, meaning the diagnosis of the disease.
Due to how a NN does not actually process words and characters, it can instead only process numerical data. As a result, the name of each disease could not simply be utilized for prediction, rather an encoder had to be utilized. As for this component, I relied on manual encoding, meaning each of the 42 diseases discussed in the dataset were given a unique number. Furthermore, every time the model made a prediction, it would thus be outputting a numerical representation of the disease. However, the conversion from this number to the respective disease was quite simple.

### Dependencies
Tensorflow (Keras API) = 2.11.0
Numpy = 1.21.5
Pandas = 1.4.4
SciKitLearn = 1.0.2
