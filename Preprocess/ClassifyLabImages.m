function ClassifyLabImages()
clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.


% Load challenge Training data
load("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat");

% Provide the path to the input images, for example
% 'C:\AGC_Challenge_2019\images\'
imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";

outputImageRootFolder= "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/TRAINING/";
% Take images for each folder named by the labe_id

% Process all images in the Training set

for j = 1:length(AGC19_Challenge3_TRAINING)
    A = imread( sprintf('%s%s',...
        imgPath, AGC19_Challenge3_TRAINING(j).imageName ));
    
    % Get label ID from training set
    label_id = AGC19_Challenge3_TRAINING(j).id;
    strLabelId = num2str(label_id);
    % Prepare new image name to write as output image
    
    
    % Write image file, if the folder does not exist, create folder by
    % label_id
    writeImageToFilePathName = outputImageRootFolder + strLabelId + "/" + AGC19_Challenge3_TRAINING(j).imageName;
    % writeImageToFilePath = writeImageToFilePathName + "." + exportFileType;
    
    
    newSubFolder = outputImageRootFolder + strLabelId;
    if ~exist((newSubFolder), 'dir')
        mkdir(newSubFolder);
    end
    imwrite(A,writeImageToFilePathName,'jpg');
    fprintf("processing image: %0.0f\n", j);
    
end
end

