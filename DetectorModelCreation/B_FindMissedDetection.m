clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.


% Confirm detector model and what wrong in detection



% Load challenge Training data
load("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat")

% Provide the path to the input images, for example
% 'C:\AGC_Challenge_2019\images\'
imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";


% Set best parameter that obtain from searching good parameter combination
setGlobalDetector(10,[85,85]);

trainingImages = [];
trainingLabels = [];
boxesSize = [];
imagesSize = [];

missedDetectName = [];

diary log_missed_detection
% Process all images in the Training set
cntMissed = 0;
for j = 1 : length( AGC19_Challenge3_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC19_Challenge3_TRAINING(j).imageName ));

    % defect face from image by our detector model.
    [bboxes] = MyFaceDetectionFunction(A);

    
    % Check our detector model performance if it detects correctly
    % If not, we need to find the way to improve this.
    if isempty(bboxes) ~= isempty(AGC19_Challenge3_TRAINING(j).faceBox)
        %fprintf("Wrong detection %0.0f\n", j);
        str = AGC19_Challenge3_TRAINING(j).imageName ...
            + " - " + isempty(bboxes)...
            + " - " + isempty(AGC19_Challenge3_TRAINING(j).faceBox);
        fprintf("%s\n", AGC19_Challenge3_TRAINING(j).imageName);
        missedDetectName = [missedDetectName; str];
        cntMissed = cntMissed + 1;
    end
     
end
fprintf("Total missed: %0.0f\n", cntMissed);

diary off;





