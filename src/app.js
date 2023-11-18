import { join } from 'path';
import appRootPath from 'app-root-path';
import { moments, tensor } from '@tensorflow/tfjs-node';
import loadCSV from './utils/csv-loader.js';

// Load the CSV file path
const csvFilePath = join(appRootPath.path, 'data', 'kc_house_data.csv');

// Define options for loading CSV data
const loadOptions = {
  dataColumns: ['lat', 'long', 'sqft_lot'], // Columns representing features
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
  // Compute mean and variance of the features
  const { mean, variance } = moments(features, 0);

  // Standardize the prediction point using mean and variance
  const scaledPredictionPoint = predictionPoint.sub(mean).div(variance.pow(0.5));

  // Standardize features and the prediction point
  const scaledFeatures = features.sub(mean).div(variance.pow(0.5));
  const scaledDistances = scaledFeatures.sub(scaledPredictionPoint).pow(2).sum(1).pow(0.5);

  // Combine distances with labels, sort by distance, and take the top 'k' neighbors
  const topK = scaledDistances
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
    .slice(0, k);

  // Calculate the predicted value by averaging the 'k' nearest neighbors' labels
  const prediction = topK.reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k;

  return prediction;
};

// Iterate through each test feature and make predictions using KNN
testFeatures.forEach((testPoint, i) => {
  const prediction = knn(tensorFeatures, tensorLabels, tensor(testPoint), 10);

  // Calculate the error percentage
  const error = ((testLabels[i][0] - prediction) / testLabels[i][0]) * 100;

  // Display predictions and error percentage
  if (i === 0) {
    console.log('\n\n+--------- Predictions -----------+\n');
  }
  console.log(`  [+] Predicted Price: $${prediction}`);
  console.log(`      Actual Price: $${testLabels[i][0]}`);
  console.log(`      Error: ${error.toFixed(2)}%\n`);

  if (i === testFeatures.length - 1) {
    console.log('+---------------------------------+\n\n');
  }
});
