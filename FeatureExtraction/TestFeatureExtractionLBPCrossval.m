clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.


imds = imageDatastore("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Grayscale128-Smooth/", ...
    'IncludeSubfolders',true,'FileExtensions','.jpg', "LabelSource", "foldernames");


[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.


[features, labels] = helperExtractLBPFeatures(imds);
 
% cumluative sum
% [coeff,score,latent,tsquared,explained,mu] = pca(features,  'Centered', true);
% plotCumsumPCA(explained);

% nComponents = 10;
% [coeff,score,latent,tsquared,explained,mu] = pca(features, ...
%     'NumComponents', nComponents, 'Centered', true); 
% 
% dataProjected = projectData(features, coeff);

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(features, labels);

CVMdl = crossval(classifier, 'KFold', 3);
loss = kfoldLoss(CVMdl);
fprintf('CV: %.2f\n', (1 - loss)*100);