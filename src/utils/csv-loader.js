import fs from 'fs';
import _ from 'lodash';
import shuffleSeed from 'shuffle-seed';

/**
 * Extracts specific columns from the given data.
 *
 * @param {Array<Array<any>>} data - The dataset containing rows and columns.
 * @param {Array<string>} columnNames - Names of columns to be extracted.
 * @returns {Array<Array<any>>} - Extracted columns from the dataset.
 */
const extractColumns = (data, columnNames) => {
  // Get the header row from the dataset
  const headers = _.first(data);

  // Get indexes of columns to be extracted
  const indexes = _.map(columnNames, (column) => headers.indexOf(column));

  // Extract columns based on indexes
  const extracted = _.map(data, (row) => _.pullAt(row, indexes));

  return extracted;
};

/**
 * Loads a CSV file, processes the data, and performs optional operations like shuffling and splitting.
 *
 * @param {string} filename - The name of the CSV file to load.
 * @param {Object} options - Additional options for data processing.
 * @param {Array<string>} options.dataColumns - Columns representing the data.
 * @param {Array<string>} options.labelColumns - Columns representing the labels.
 * @param {Object} options.converters - Converters for specific column data.
 * @param {boolean} options.shuffle - Flag to shuffle the data.
 * @param {boolean|number} options.splitTest - Flag to split data for testing. If number, defines the test set size.
 * @returns {Object} - Processed dataset features and labels.
 */
export default (
  filename,
  { dataColumns = [], labelColumns = [], converters = {}, shuffle = false, splitTest = false },
) => {
  // Read the CSV file
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });

  // Parse the CSV data into a 2D array
  data = _.map(data.split('\n'), (row) => row.split(','));

  // Remove trailing empty rows
  data = _.dropRightWhile(data, (val) => _.isEqual(val, ['']));
  const headers = _.first(data);

  // Process data rows
  data = _.map(data, (row, index) => {
    // Skip processing of the header row
    if (index === 0) {
      return row;
    }

    // Process each element in the row
    return _.map(row, (element, colIndex) => {
      // Apply converters if defined for the specific column
      if (converters[headers[colIndex]]) {
        const converted = converters[headers[colIndex]](element);
        return _.isNaN(converted) ? element : converted;
      }

      // Convert string numbers enclosed in quotes to numbers
      const result = parseFloat(element.replace('"', ''));
      return _.isNaN(result) ? element : result;
    });
  });

  // Extract label and data columns
  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);

  // Remove header row from extracted data
  data.shift();
  labels.shift();

  // Shuffle data if specified
  if (shuffle) {
    data = shuffleSeed.shuffle(data, 'phrase');
    labels = shuffleSeed.shuffle(labels, 'phrase');
  }

  // Split data for testing if specified
  if (splitTest) {
    const trainSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length / 2);
    return {
      features: data.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: data.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize),
    };
  } else {
    return { features: data, labels };
  }
};
