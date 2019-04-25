clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

%matPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat";
%imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";
%[trainingImages, trainingLabels] = getTrainingData(matPath, imgPath);

% fprintf(mat2str(size(trainingImages)),mat2str(size(trainingLabels)));

% =============================================================
% ====== Produce processed images from our training set
% =============================================================
sourceImageRootFolder = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Preprocess/NewTrainImages/Orignial/";

% !!! Select mode of image, 1: grayscale, 3: RGB
imageMode = 1;
% !!! Change target folder according to selected image mode (Grayscale or RGB)
% outputImageRootFolder= "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/GrayscaleFilterRotate128/";
outputImageRootFolder= "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Grayscale128/";
%outputImageRootFolder = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Preprocess/NewTrainImages/Processed/";

outputFailedImageRootFolder = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Preprocess/NewTrainImages/Failed/";
% Create processed and failed folders name if they don't exist.
if ~exist((outputImageRootFolder), 'dir')
        mkdir(outputImageRootFolder);
end
if ~exist((outputFailedImageRootFolder), 'dir')
        mkdir(outputFailedImageRootFolder);
end

% === Set detector model parameter
% Set best parameter that obtain from CrossValidationParameterSearching
setGlobalDetector(5,[85,85]);


% !!! Set isCalcWidthHeight
% isCalcWidthHeight = 0, when boundar box is directly returned from detector
% isCalcWidthHeight = 1, when boundary box is directly returned from
% TrainingSet provided by professor
isCalcWidthHeight = 0;

% !!! Set array of image size for width and height
% Different pre-trained model has different requirement
imageSizes = [128 128];
PreprocessingImages(sourceImageRootFolder, outputImageRootFolder, ...
    outputFailedImageRootFolder, imageMode, isCalcWidthHeight, imageSizes);

% !!!After this you can take images from target output folder to /DS/ folder