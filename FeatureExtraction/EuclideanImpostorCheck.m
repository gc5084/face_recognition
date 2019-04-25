clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

diary log_result_hog_euc_impostor
load '/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/myFaceRecognitionModelMergedGray128.mat';

impostorPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/Impostor/";

imdsImpostor = imageDatastore(impostorPath, ...
        'IncludeSubfolders',false,'FileExtensions',{'.jpg'});

    
impostor_group_distances = [];
 %for impIdx = 1:80 % Internal group
 for impIdx = 1:80    % impostor check
    
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
        for i = 1:length(imdsImpostor.Files)
            queryImage = imread(imdsImpostor.Files{i});
            queryImage = imbinarize(queryImage);
            queryImageFeature = extractHOGFeatures(queryImage,'CellSize',cellSize);
            qVector = projectData(queryImageFeature, coeff);
    
            % Mahalanobis distance in predicted label space.
            %d = sqrt((qVector-mu)*inv(Cov)*(qVector-mu)');
            d = pdist([mu; qVector],'euclidean');
            % Row matrix
            dist = [dist d];
            fprintf('Impostor: %s\tGroup: %.0f\tDistance: %.2f\n', ...
                imdsImpostor.Files{i}, impIdx, d);
        end
    
        fprintf('Group: %s Min: %.2f Max: %.2d\n', num2str(impIdx), min(dist), max(dist));
        impostor_group_distances = [impostor_group_distances; [impIdx min(dist) max(dist) mu]];

    
end

diary off