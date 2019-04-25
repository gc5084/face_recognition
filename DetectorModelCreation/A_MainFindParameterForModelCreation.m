clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.
% Basic script for Face Detection Challenge
% --------------------------------------------------------------------
% AGC Challenge 2019
% Universitat Pompeu Fabra
%

% Load challenge Training data
load("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat")

% Provide the path to the input images, for example
% 'C:\AGC_Challenge_2019\images\'
imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";


diary result_log

% thresholds = [8; 10; 12; 14; 16; 18; 20; 22; 24; 26];
% minSize by default is 20 and we can't set it less than 20 so we start from 20
% minSize = [20; 25; 30; 35; 40; 45; 50; 55; 60; 65; 70; 75; 80; 85; 90; 95; 100; 105; 110; 115; 120; 125; 130; 135; 140; 145; 150];

thresholds = [10];
minSize = [85];
% Number of fold for cross validation
KFoder = 5;
numberOfTotalImages = size(AGC19_Challenge3_TRAINING,2);
cvIndices = crossvalind('Kfold',numberOfTotalImages,KFoder);


lenThreshold = size(thresholds,1);
lenMinSize = size(minSize,1);
for thersholdIdx = 1:lenThreshold
    for minsizeIdx = 1:lenMinSize
        
        % Combine parameters
        currentMinSize = minSize(minsizeIdx);
        currentThreshold =  thresholds(thersholdIdx);
        detector = vision.CascadeObjectDetector('MergeThreshold', currentThreshold ...
             ,'MinSize', [currentMinSize, currentMinSize]);
      
        
        % Train & Test data preparation
        sumFdScore = 0;
        for k = 1:KFoder
            
            % Initialize results structure
            DetectionSTR = struct();
            imagesData = AGC19_Challenge3_TRAINING(cvIndices==k);
            lenThreshold = size(thresholds,1);
            lenMinSize = size(minSize, 1);
            
            
            % Initialize timer accumulator
            total_time = 0;
            % Process all images in the Training set
            for j = 1 : length( imagesData )
                A = imread( sprintf('%s%s',...
                    imgPath, imagesData(j).imageName ));
                
                try
                    % Timer on
                    tic;
                    
                    % ###############################################################
                    % Your face detection function goes here. It must accept a single
                    % input parameter (the input image A) and it must return one or
                    % more bounding boxes corresponding to the facial images found
                    % in image A, specificed as [x1 y1 x2 y2]
                    % Each bounding box that is detected will be indicated in a
                    % separate row in det_faces
                    
                    det_faces = DetectBoundingBoxForModelCreation( detector, A );

                    % ###############################################################
                    
                    % Update total time
                    tt = toc;
                    total_time = total_time + tt;
                    
                catch
                    % If the face detection function fails, it will be assumed that no
                    % face was detected for this input image
                    det_faces = [];
                end
                
                % Store the detection(s) in the resulst structure
                DetectionSTR(j).det_faces = det_faces;
            end
            
            % Compute detection score
            FD_score = CHALL_AGC19_ComputeDetScores(...
                DetectionSTR, imagesData, 0);

            
            % Display summary of results
            fprintf(1, 'F1-score: \t%.2f%% \tTime in seconds \t%0f \t Total time: %dm %ds\n', ...
                100 * FD_score, ...
                total_time, ...
                int16( total_time/60),...
                int16(mod( total_time, 60)) );
            
            % end of this round, add up the FD score
            sumFdScore = sumFdScore + FD_score; 
        end
        % end of all K-fold then find average value
        average = sumFdScore/KFoder;
        fprintf("MergeThreshold \t%0d \t MinSize \t%0d Avg: %0.2f \n", currentThreshold , currentMinSize, average*100);
    end

end


diary off