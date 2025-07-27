const {Neurex, CsvDataHandler, Interpreter, OneHotEncoded, split_dataset, ClassificationMetrics, IntegerLabeling} = require('neurex');

const model = new Neurex();
const csv = new CsvDataHandler();
const interpreter = new Interpreter();

const dataset = csv.read_csv('iris-dataset.csv');
const extract_column = csv.extractColumn('iris', dataset);
const features = csv.normalize('MinMax',csv.rowsToInt(dataset));

const labels = IntegerLabeling(extract_column);
const {X_train, Y_train, X_test, Y_test} = split_dataset(features, labels, 0.2);

model.configure({
    optimizer: 'adam',
    randMin: -0.1,
    randMax: 0.1,
    learning_rate: 0.0001
});

model.inputShape(X_train);
model.construct_layer("relu", 32);
model.construct_layer("relu", 16);
model.construct_layer("relu", 8);
model.construct_layer("softmax", 3);

model.train(X_train, Y_train, "sparse_categorical_cross_entropy", 5000, 32);
model.saveModel('sparse_test3');
interpreter.loadSavedModel('sparse_test3.nrx');
const predictions = interpreter.predict(X_test);

ClassificationMetrics(predictions, Y_test, "sparse_categorical", ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]);