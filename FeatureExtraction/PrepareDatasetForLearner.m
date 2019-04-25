clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Grayscale128-Smooth/";
% imgPath = "/Volumes/Work/UPF/Class_FACIAL/LAB2-FG/CKDB/";
imds = imageDatastore(imgPath, ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg'}, "LabelSource", "foldernames");



[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

trainImgMatrix = [];
rowTrainSet = length(imds.Files);
for r = 1: rowTrainSet
    img = imread(imds.Files{r});
    imgVector = reshape(img, [1 size(img,1)*size(img,2)]);
    trainImgMatrix = [trainImgMatrix; single(imgVector)];
end

Labels = imds.Labels;


% cumluative sum
% [coeff,score,latent,tsquared,explained,mu] = pca(trainImgMatrix,  'Centered', true);
% plotCumsumPCA(explained);

nComponents = 500;
[coeff,score,latent,tsquared,explained,mu] = pca(trainImgMatrix, ...
    'NumComponents', nComponents, 'Centered', true); 

dataProjected = projectData(trainImgMatrix, coeff);

column = dataProjected;
X = array2table(column);

X.Response =  categorical(imds.Labels);
X = movevars(X,'Response','Before','column1');
head(X,5)