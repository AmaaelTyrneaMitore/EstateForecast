import { join } from 'path';
import appRootPath from 'app-root-path';

import loadCSV from './utils/csv-loader.js';

const filename = join(appRootPath.path, 'data', 'kc_house_data.csv');

const options = {
  dataColumns: ['lat', 'long'],
  labelColumns: ['price'],
  shuffle: true,
  splitTest: 5,
};

const { features, labels, testFeatures, testLabels } = loadCSV(filename, options);

console.log('Features:', features);
console.log('Labels:', labels);

if (testFeatures && testLabels) {
  console.log('Test Features:', testFeatures);
  console.log('Test Labels:', testLabels);
}
