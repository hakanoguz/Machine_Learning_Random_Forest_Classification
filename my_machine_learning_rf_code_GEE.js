// ************************************************************************
// MACHINE LEARNING (RANDOM FOREST) SUPERVISED CLASSIFICATION WITH LANDSAT 9
// This javascript code runs in Google Earth Engine
// This script classifies Landsat 9 Collection 2 Level 2 Tier 2 Imagery
// Modified by Hakan OGUZ on November 7th, 2022
// ************************************************************************

// Step 1
// Load Landsat 9 data

var dataset = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
	.filterDate('2022-01-01', '2022-05-30')
	.filterBounds(roi)
	.filterMetadata('CLOUD_COVER', 'less_than', 10);

// Applies scaling factors

function applyScaleFactors(image) {
	var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
	var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
	return image.addBands(opticalBands, null, true)
				.addBands(thermalBands, null, true);
}

var rescale = dataset.map(applyScaleFactors);
var image = rescale.median();

var visualization = {
	bands: ['SR_B4', 'SR_B3', 'SR_B2'],
	min: 0.0,
	max: 0.3,
};
Map.addLayer(image.clip(roi), visualization, 'Landsat 9');
Map.centerObject(roi, 10);

// Step 2
// Create training data

var = training = Barren.merge(Forest).merge(Water).merge(Urban).merge(Crop);
var label = 'Class';
var bands = ['SR_B1', 'SR_B2', 'SR_B3','SR_B4', 'SR_B5', 'SR_B7'];
var input = image.select(bands);

// Overlay the pointson the imagery to get training

var trainImage = input.sampleRegions({
	collection: training,
	properties: [label],
	scale: 30
});

print(trainImage);

var trainingData = trainImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lessThan('random', 0.8));
var testSet = trainingData.filter(ee.Filter.greaterThanOrEquals('random', 0.8));

// Step 3
// Classification Model
// Make a random forest classifier and train it

var classifier = ee.Classifier.smileRandomForest(10)
	.train({
		features: trainSet,
		classProperty: label,
		inputProperties: bands
	});

// Classify the image

var classified = input.classify(classifier);

// Define a color palette for the classification
var landcoverPalette = [
	'#0c2c84', //water (0)
	'#e31a1c', //urban (1)
	'#005a32', //forest (2)
	'#FF8000', //crop (3)
	'#969696', //barren (4)
	];
	Map.addLayer(classified.clip(roi), {palette: landcoverPalette, min: 0, max: 4 }, 'classification');

// Step 4
// Accuracy Assessment
// Classify the testingSet and get a confusion matrix

var confusionMatrix = ee.ConfusionMatrix(testSet.classify(classifier)
	.errorMatrix({
		actual: 'Class',
		predicted: 'classification'
	}));

print('Confusion matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Producers Accuracy:', confusionMatrix.producersAccuracy());
print('Consumers Accuracy:', confusionMatrix.consumersAccuracy());

// Step 5
// Export classified map to Google Drive

Export.image.toDrive({
	image: classified,
	description: 'Landsat_Classified_RF',
	scale: 30,
	region: roi,
	maxPixels: 1e13
});


