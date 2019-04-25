clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

diary log_result_hog_mahal
load '/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/myFaceRecognitionModelMergedGray128.mat';

impostorPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/Impostor/";

imdsImpostor = imageDatastore(impostorPath, ...
        'IncludeSubfolders',false,'FileExtensions',{'.jpg'});

% impostorPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/ID/13";
% imdsImpostor = imageDatastore(impostorPath, ...
%         'IncludeSubfolders',false,'FileExtensions',{'.jpg'});

    
group_distances = [];
 for impIdx = 1:80 % Internal group

   
    imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/AugmentedGrayscale128/" + num2str(impIdx) + "/";
   
    imdsPredictedLabel = imageDatastore(imgPath, ...
        'IncludeSubfolders',true,'FileExtensions',{'.jpg'});
    
    % Each face space
    imgTemp = imread(imdsPredictedLabel.Files{1});
    [hog, vis] = extractHOGFeatures(imgTemp,'CellSize',cellSize);
    hogFeatureSize = length(hog);
    [X, ~] = helperExtractHOGFeatures(imdsPredictedLabel, hogFeatureSize, cellSize);
    subSpaceOfPredictedLabel = projectData(X, coeff);
    mu = mean(subSpaceOfPredictedLabel,1);
    Cov = cov(subSpaceOfPredictedLabel);
        
        dist = [];
        for i = 1:length(imdsPredictedLabel.Files)
            queryImage = imread(imdsPredictedLabel.Files{i});
            queryImage = imbinarize(queryImage);
            queryImageFeature = extractHOGFeatures(queryImage,'CellSize',cellSize);
            qVector = projectData(queryImageFeature, coeff);
    
            % Mahalanobis distance in predicted label space.
            %d = sqrt((qVector-mu)*inv(Cov)*(qVector-mu)');
            d = pdist([mu; qVector],'euclidean');
            % Row matrix
            dist = [dist d];
            fprintf('Distance: %.2f\n', d);
        end
    
        fprintf('Group: %s Min: %.2f Max: %.2d\n', num2str(impIdx), min(dist), max(dist));
        group_distances = [group_distances; [impIdx min(dist) max(dist) mu]];

    
end

diary off