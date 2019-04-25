clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/MixGray128/";
% imgPath = "/Volumes/Work/UPF/Class_FACIAL/LAB2-FG/CKDB/";
imds = imageDatastore(imgPath, ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg','.tiff'}, "LabelSource", "foldernames");

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.



% Tunable parameter
cellSize = [13 13];

imgTemp = imread(imds.Files{1});
[hog, vis] = extractHOGFeatures(imgTemp,'CellSize',cellSize);
hogFeatureSize = length(hog);
numImages = numel(trainingSet.Files);


% ====
% Get labels for each image.
[trainingFeatures, trainingLabels] = helperExtractHOGFeatures(trainingSet, hogFeatureSize, cellSize);


% cumluative sum
% [coeff,score,latent,tsquared,explained,mu] = pca(trainingFeatures,  'Centered', true);
% plotCumsumPCA(explained);

diary log_result_hog_lda
% 10 50 100 150 200 300 400 500 600 650 700 750 800
nComponents = [500];
for n=1:length(nComponents)
    
[coeff,score,latent,tsquared,explained,mu] = pca(trainingFeatures, ...
    'NumComponents', nComponents(n), 'Centered', true); 

dataProjected = projectData(trainingFeatures, coeff);


% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
% myFRModel = fitcecoc(dataProjected, trainingLabels);
myFRModel = fitcdiscr(dataProjected,trainingLabels);

%save 'myFaceRecognitionModel.mat' 'myFRModel';

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures, testLabels] = helperExtractHOGFeatures(testSet, hogFeatureSize, cellSize);

testDataProjected = projectData(testFeatures, coeff);

% Make class predictions using the test features.
[predictedLabels, scores] = predict(myFRModel, testDataProjected);

% Calc accuracy
accuracy = calcAccuracy(testLabels, predictedLabels);
% fprintf('PCA %.0f LDA: %.2f\n', nComponents(n),  accuracy);
CVMdl = crossval(myFRModel, 'KFold', 3);
loss = kfoldLoss(CVMdl);
fprintf('PCA %.0f LDA: %.2f CV: %.2f\n', nComponents(n),  accuracy, (1 - loss)*100);
end

% save '/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/myFaceRecognitionModel10pp.mat' ...
%     'coeff' 'myFRModel' 'nComponents' 'cellSize' 'imageSize';

diary off;