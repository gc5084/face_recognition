% clc;        % Clear the command window.
% close all;  % Close all figures (except those of imtool.)
% clear;      % Erase all existing variables. Or clear vars.
% workspace;  % Make sure the workspace panel is showing.

% Access the trained model
% vgg16 requires Deep Learning Toolbox Model
net = vgg16;

% See details of the architecture 
disp(net.Layers);


% Task - Plot the weights of the first convolutional layer 
% as it is done in the example and explain intuitively 
% which kind of features do you think that the network is extracting.

% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. 
% There are 96 individual sets of weights in the first layer.
% figure
% montage(w1)
% title('First convolutional layer weights')



%-----------------------------------------------------------------%
%-------- Part 2: Train CKDB dataset, and report accuracy  -------%
%-----------------------------------------------------------------%


% -------- Dataset ------%

% Load CKDB dataset 
% Use splitEachLabel method to trim the set.
% imds = imageDatastore("./CKDB/", ...
%     'IncludeSubfolders',true,...
%     'FileExtensions','.tiff',...
%     'LabelSource','foldernames');

% TODO check RGB and grayscale
% Alexnet was trained witn RGB
imagePaht = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/RGB224/";
imds = imageDatastore(imagePaht, ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg','.jpeg'}, ...
    "LabelSource", "foldernames");

% https://uk.mathworks.com/help/deeplearning/ref/augmentedimagedatastore.html
augmenter = imageDataAugmenter( ...
    'RandRotation',[-20 20], ...
    'RandXScale',[1 5],...
    'RandXScale',[1 5],...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

% https://uk.mathworks.com/help/deeplearning/ref/augmentedimagedatastore.html
imageSize = [224 224 3];
augimds = augmentedImageDatastore(imageSize,imds,...
    'DataAugmentation',augmenter); %  



% -------- Balance dataset ------%
% Balance data in training set
% Because imds above contains an unequal number of images per category, 
% Let's first adjust it, so that the number of images 
% in the training set is balanced. 
% determine the smallest amount of images in a category
%tbl = countEachLabel(imds);
%minSetCount = min(tbl{:,2}); 
% Use splitEachLabel method to trim the set.
%imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
%countEachLabel(imds)


% -------- Split dataset for train and test ------%

% Prepare Training and Test Image Sets
% Below number is ratio between training and testing
% Specify training ratio here.
[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
%numTrainFiles = size(imds.Files, 1);
% Notice that each set now has exactly the same number of images.
%countEachLabel(imds)

% -------- Check CNN ------%
% Check input size of image for CNN
imageSize = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)))

% -------- Convert dataset for using CNN ------%
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(81)
    softmaxLayer
    classificationLayer];

opts = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',testSet);

net = trainNetwork(augimds,layers,opts);
save 'vgg16net.mat' 'net'
%augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
%augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');





% -------- Extract training features ------% 
% Typically starting with the layer right before the classification layer 
% is a good place to start.

featureLayer = 'fc7';
featuresTrain = activations(net,trainingSet,featureLayer,'OutputAs','rows');
featuresTest = activations(net,testSet,featureLayer,'OutputAs','rows');


% -------- Classification ------% 
% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
cnnModelLab = fitcecoc(featuresTrain,trainingSet.Labels);
save 'vgg16net_extdb_svm.mat' 'cnnModelLab'
% save 'cnnModelLab.mat' 'cnnModelLab'

predictedLabels = predict(cnnModelLab,featuresTest);


% -------- Accuracy Score ------%
accuracy = calcAccuracy(testSet.Labels,predictedLabels);


