const {Neurex, CsvDataHandler, MinMaxScaler, split_dataset, Interpreter, RegressionMetrics} = require('neurex');

    const csv = new CsvDataHandler();
    const model = new Neurex();
    const interpreter = new Interpreter();
    const scaler = new MinMaxScaler();

    const dataset = csv.read_csv('housing.csv');
    // csv.tabularize(dataset); // to view

    const adjusted_dataset = csv.removeColumns(["ZN", "CHAS"], dataset); // remove columns that might you don't need
    // csv.tabularize(adjusted_dataset); // to view

    const formatted_dataset = csv.rowsToInt(adjusted_dataset); // converts string cell values to numbers

    const extractedColumn = csv.extractColumn("MEDV",formatted_dataset);
    // csv.tabularize(formatted_dataset); // check the formatted_dataset, it mutated and the MEDV column is extracted. Try logging the extracted column

    // =========== normalization ============= //
    scaler.fit(formatted_dataset);
    const features = scaler.transform(formatted_dataset); // normalizes the features to values between 0 to 1
    scaler.fit(extractedColumn);
    const target = scaler.transform(extractedColumn); // normalizes the target values between 0 to 1

    // ============ splitting the dataset into training and test sets ============ //
    const {X_train, Y_train, X_test, Y_test} = split_dataset(features, target, 0.2); // 0.2 is the test size, so the remaining 0.8 (80%) belongs to the training sets

    // ============ building the network ============ //

    model.inputShape(X_train);
    model.construct_layer("relu", 10);
    model.construct_layer("relu", 8);
    model.construct_layer("relu", 5);
    model.construct_layer("relu", 4);
    model.construct_layer("relu", 3);
    model.construct_layer("linear", 1); // <- output layer 

    model.configure({
        optimizer: 'adam', // we'll use SGD optimizer
        learning_rate: 0.001,
        // these are other configurable options
        // randMin: -0.1, // minimum range for initializing weights and biases
        // randMax: 0.1 // maximum range for initializing weights and biases
    });

    model.train(X_train, Y_train, 'mse', 15000, 12); // train the network
    model.saveModel('housing'); // will be save as housing.nrx

    const predictions = model.predict(X_test); // or use interpreter.predict() after you load the nrx model using interpreter.loadSavedModel('housing.nrx');

    RegressionMetrics(predictions, Y_test);