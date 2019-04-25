clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

% matPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat";
% imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";
% resizeSize = 227;
%[trainingImages, trainingLabels] = getGrayscaleTrainingData(matPath, imgPath, resizeSize, resizeSize);
% Training take  time, check more feature extraction method


imds = imageDatastore("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Grayscale224-Smooth/", ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg'}, "LabelSource", "foldernames");


[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.



[trainingFeatures, trainingLabels] = helperExtractSURFFeatures(trainingSet);
 
% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels);

[testFeatures, testLabels] = helperExtractSURFFeatures(testSet);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

accuracy = calcAccuracy(testLabels, predictedLabels);

CVMdl = crossval(classifier, 'KFold', 5);
loss = kfoldLoss(CVMdl);
disp((1-loss)*100);
