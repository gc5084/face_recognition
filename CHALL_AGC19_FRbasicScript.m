clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

% Basic script for Face Recognition Challenge
% --------------------------------------------------------------------
% AGC Challenge 2019 
% Universitat Pompeu Fabra
%

% Load challenge Training data
load AGC19_Challenge3_Training.mat

% Provide the path to the input images, for example 
% 'C:\AGC_Challenge_2019\images\'
imgPath = [];

% Initialize results structure
AutoRecognSTR = struct();

% Initialize timer accumulator
total_time = 0;

% Load Face Recognition model from myFaceRecognitionModel.mat
% This file must contain a single variable called 'myFRModel'
% with any information or parameter needed by the
% function 'MyFaceRecognFunction' (see below)

% Load the model.
%modelFile = 'myFaceRecognitionModelClearnMergedGrayscale128.mat';
modelFile = 'googleImgRot.mat';
load(modelFile);
myFRModel = my_FRmodel;
% modelFile = 'myFaceRecognitionModel.mat';



% myFRModel = struct('classifier', myFRModel, ...
%     'faceDetector', vision.CascadeObjectDetector('MergeThreshold', 5, 'MinSize', [85 85]), ...
%     'imageSize', [128 128], 'cellSize', [13 13], ...
%     'nComponents', 500, 'pcaCoeff', coeff);
% 
% save 'myFaceRecognitionModel.mat' 'myFRModel'

diary log_myFaceRecognitionModelBeforeInpostorCheck
results = [];
% Process all images in the Training set
for j = 1 : length( AGC19_Challenge3_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC19_Challenge3_TRAINING(j).imageName ));    
    
    %try
        % Timer on
        tic;
                
        % ###############################################################
        % Your face recognition function goes here. It must accept 2 input
        % parameters:
        %
        % 1. the input image A
        % 2. the recognition model
        %
        % and it must return a single integer number as output, which can
        % be:
        % a) A number between 1 and 80 (representing one of the identities
        % in the trainig set)
        % b) A "-1" indicating that none of the 80 users is present in the
        % input image
        %

        autom_id = my_face_recognition_function( A, myFRModel );
       % fprintf('Processed image name %s\t', AGC19_Challenge3_TRAINING(j).imageName);
       
       if AGC19_Challenge3_TRAINING(j).id ~= autom_id
       fprintf('Img: %s\tLabel: %.0f\tPredict: %.0f\tResult: %.0f\n', ...
           AGC19_Challenge3_TRAINING(j).imageName, ...
           AGC19_Challenge3_TRAINING(j).id, autom_id, (AGC19_Challenge3_TRAINING(j).id == autom_id));
       end
       results = [results; [ ...
            AGC19_Challenge3_TRAINING(j).id autom_id (AGC19_Challenge3_TRAINING(j).id == autom_id)]];
%         if autom_id ~= -1 
%             if AGC19_Challenge3_TRAINING(j).id == autom_id
%             fprintf('Predict Face Correct: %s', AGC19_Challenge3_TRAINING(j).imageName);
%             cntPredictCorrect = cntPredictCorrect + 1;
%             else
%                 
%             end
%             
%         end
        % ###############################################################
        
        % Update total time
        tt = toc;
        total_time = total_time + tt;
        
    %catch
        % % If the face recognition function fails, it will be assumed that no
        % % user was detected for this input image
        % autom_id = -1;
    %end

    % Store the detection(s) in the resulst structure
    AutoRecognSTR(j).id = autom_id;
end
   
% Compute detection score
FR_score = CHALL_AGC19_ComputeRecognScores(...
    AutoRecognSTR, AGC19_Challenge3_TRAINING);

% Display summary of results
fprintf(1, '\nF1-score: %.2f%% \t Total time: %dm %ds\n', ...
    100 * FR_score, int16( total_time/60),...
    int16(mod( total_time, 60)) );
diary off