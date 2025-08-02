const { Neurex, Layers } = require("neurex");

// Create instances of the Neurex and Layers classes
const model = new Neurex();
const layer = new Layers();

// 1. Define the network architecture
// The XOR problem is not linearly separable, so a hidden layer is required.
model.sequentialBuild([
    // Input layer with 2 neurons for the two XOR inputs
    layer.inputShape({ features: 2 }),
    
    // A hidden layer with 4 neurons and a ReLU activation function
    layer.connectedLayer("relu", 4),
    
    // The output layer with 1 neuron and a sigmoid activation for binary classification
    layer.connectedLayer("sigmoid", 1)
]);

// 2. Build the model to initialize weights and biases
model.build();

// 3. Configure the model's training parameters
// Using the Adam optimizer with a learning rate of 0.01
model.configure({
    optimizer: 'adam',
    learning_rate: 0.01
});

// Display the model summary to confirm the architecture
model.modelSummary();

// 4. Prepare the training data for the XOR problem
const trainX = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

// The expected outputs (ground truth)
const trainY = [
    [0],
    [1],
    [1],
    [0]
];

// 5. Train the model
// Use binary cross-entropy loss, train for 1000 epochs, and use a batch size of 2
// The task will be automatically inferred as 'binary_classification'
model.train(trainX, trainY, 'binary_cross_entropy', 1000, 2);

// 6. Make predictions after training
const predictions = model.predict(trainX);

// 7. Print the results
console.log("XOR Predictions:");
trainX.forEach((input, i) => {
    // The sigmoid output is a probability. Convert it to a binary prediction (0 or 1).
    const predictedValue = predictions[i][0] > 0.5 ? 1 : 0;
    console.log(`Input: [${input}] -> Predicted: ${predictedValue} (Raw Output: ${predictions[i][0].toFixed(4)}) | Actual: ${trainY[i][0]}`);
});