clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

% ========== Take train test dataset ========

% ========== Split data ========
% !! ========== consider to balance data ========


% ========== Train model ========

 s = 128;
 imgSizes = [s s];
 
% % PCA
%  matFilePath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat";
%  imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";
% % 
% 
% [trainingImages, trainingLabels] = getGrayscaleTrainingData(matFilePath, imgPath, imgSizes);
% 
% reshapedTrainingImages = [];
% for i=1:length(trainingImages)
%     reshapedImg  = reshape(trainingImages(i,:,:),1, s*s);
%     reshapedTrainingImages(i,:) = reshapedImg;
% end 


imds = imageDatastore("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Grayscale128/", ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg', '.jpeg'}, "LabelSource", "foldernames");

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');

reshapedTrainingImages = [];
for i=1:length(trainingSet.Files)
    img = imread(trainingSet.Files{i});
    % reshape to 1 vector
    reshapedImg  = reshape(img,1, s*s);
    reshapedTrainingImages(i,:) = reshapedImg;
end 

% cumluative sum
[coeff,score,latent,tsquared,explained,mu] = pca(reshapedTrainingImages,  'Centered', true);
plotCumsumPCA(explained);

nComponents = 600;
[coeff,score,latent,tsquared,explained,mu] = pca(reshapedTrainingImages, ...
    'NumComponents', nComponents, 'Centered', true); 

%'NumComponents', nComponents
% 'Algorithm','eig' out of memory

% find eigenvector and eigenvalue


% check eigval then we know that 2nd eigenvec explain more about data
% project all data to eigenvector
projectedData = reshapedTrainingImages * (coeff);

dataReprojected = (projectedData * coeff.') + mu;

testImgIdx = 1;
imshow(imread(trainingSet.Files{testImgIdx}));
reprojectImg = reshape(dataReprojected(testImgIdx,:), imgSizes);
imshow(uint8(reprojectImg));

% projectedData = projectData(reshapedTrainingImages,mu, coeff);
% 
% [ dataReprojected ] = reprojectData( projectedData , mu, coeff );
% LDA

% SVM
% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.

classifier = fitcecoc(projectedData, trainingSet.Labels);

%[ dataReprojected ] = reprojectData( projectedData , mu, coeff );

% Make class predictions using the test features.
predictedLabels = predict(classifier, projectedData);

accuracy = calcAccuracy(testSet.Labels, predictedLabels);
% Evalulate

