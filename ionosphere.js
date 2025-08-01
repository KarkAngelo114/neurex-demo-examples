const {Neurex, CsvDataHandler, MinMaxScaler, Interpreter, Layers, BinaryLabeling, split_dataset, ClassificationMetrics} = require('neurex');

const model = new Neurex();
const csv = new CsvDataHandler();
const scaler = new MinMaxScaler();
const interpreter = new Interpreter();
const layer = new Layers();

const dataset = csv.read_csv('ionosphere.csv');
const extracted_column = csv.extractColumn("y", dataset);
const formatted_dataset = csv.getRowElements(20, csv.rowsToInt(dataset));
const labels = BinaryLabeling(extracted_column);

const {X_train, Y_train, X_test, Y_test} = split_dataset(formatted_dataset, labels, 0.2);

scaler.fit(X_train);
const normalized_X_train = scaler.transform(X_train);
scaler.fit(X_test);
const normalized_X_test = scaler.transform(X_test);

model.configure({
    learning_rate: 0.0001,
    optimizer: 'adam',
    randMin: -0.001,
    randMax: 0.001
});

model.sequentialBuild([
    layer.inputShape({features: 20}),
    layer.connectedLayer("relu", 10),
    layer.connectedLayer("relu", 5),
    layer.connectedLayer("relu", 5),
    layer.connectedLayer("relu", 3),
    layer.connectedLayer("sigmoid", 1),
]);
model.build();

model.train(normalized_X_train, Y_train, "binary_cross_entropy",5000,12);
model.saveModel('ionosphere');
interpreter.loadSavedModel('ionosphere.nrx');
const predictions = interpreter.predict(normalized_X_test);
ClassificationMetrics(predictions, Y_test, 'binary', ["good", "bad"]);