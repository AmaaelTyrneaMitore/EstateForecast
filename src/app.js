import { join } from 'path';
import appRootPath from 'app-root-path';
import { tensor } from '@tensorflow/tfjs-node';

import loadCSV from './utils/csv-loader.js';

// Load the CSV file path
const csvFilePath = join(appRootPath.path, 'data', 'kc_house_data.csv');

// Define options for loading CSV data
const loadOptions = {
  dataColumns: ['lat', 'long'], // Columns representing features
  labelColumns: ['price'], // Column representing the label
  shuffle: true, // Shuffle the data
  splitTest: 10, // Number of rows for testing
};

// Load the CSV data using provided options
let { features, labels, testFeatures, testLabels } = loadCSV(csvFilePath, loadOptions);

// Convert loaded data into tensors
const tensorFeatures = tensor(features);
const tensorLabels = tensor(labels);

/**
 * K-Nearest Neighbors algorithm for prediction.
 *
 * @param {tf.Tensor} features - Training features.
 * @param {tf.Tensor} labels - Training labels.
 * @param {tf.Tensor} predictionPoint - Point for prediction.
 * @param {number} k - Number of neighbors to consider.
 * @returns {number} - Predicted value.
 */
const knn = (features, labels, predictionPoint, k) => {
  // Calculate distances between the prediction point and features
  const distances = features.sub(predictionPoint).pow(2).sum(1).pow(0.5);

  // Combine distances with labels, sort by distance, and take the top 'k' neighbors
  const topK = distances
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
    .slice(0, k);

  // Calculate the predicted value by averaging the 'k' nearest neighbors' labels
  const prediction = topK.reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k;

  return prediction;
};

// Make a prediction using KNN for the first test feature with k=10
const prediction = knn(tensorFeatures, tensorLabels, tensor(testFeatures[0]), 10);

// Display the prediction and actual price
console.log(`Prediction: $${prediction}\tActual Price: $${testLabels[0][0]}`);
