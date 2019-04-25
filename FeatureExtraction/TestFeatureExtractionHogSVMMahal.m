clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

diary log_result_hog_mahal
load '/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/myFaceRecognitionModelMergedGray128.mat';

impostorPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/Impostor/";

% imdsImpostor = imageDatastore(impostorPath, ...
%         'IncludeSubfolders',false,'FileExtensions',{'.jpg'});

% impostorPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/ID/13";
imdsImpostor = imageDatastore(impostorPath, ...
        'IncludeSubfolders',false,'FileExtensions',{'.jpg'});

distances = [];
for impIdx = 1:length(imdsImpostor.Files)
    
    queryImage = imread(imdsImpostor.Files{impIdx});
    imshow(queryImage);
    queryImage = imbinarize(queryImage);
    queryImageFeature = extractHOGFeatures(queryImage,'CellSize',cellSize);
    qVector = projectData(queryImageFeature, coeff);

    [label, score] = predict(myFRModel, qVector);

    labelLen = length(label);
    convertedLabel = [];
    for c = 1:labelLen
        convertedLabel = [convertedLabel,char(label(c))];
    end
    % imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/ID/";

    imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/AugmentedGrayscale128/" + convertedLabel + "/";
    
    % imgPath = "/Volumes/Work/UPF/Class_FACIAL/LAB2-FG/CKDB/";
    imdsPredictedLabel = imageDatastore(imgPath, ...
        'IncludeSubfolders',true,'FileExtensions',{'.jpg'});
    
    
    imgTemp = imread(imdsPredictedLabel.Files{1});
    [hog, vis] = extractHOGFeatures(imgTemp,'CellSize',cellSize);
    hogFeatureSize = length(hog);
    [X, ~] = helperExtractHOGFeatures(imdsPredictedLabel, hogFeatureSize, cellSize);
    
    
    subSpaceOfPredictedLabel = projectData(X, coeff);
    
    % Test Image
    % queryImage = imread("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Augmented/1/1_49.jpg");
    % queryImage = imread("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Augmented/2/2_200.jpg");
    
    %     queryImage = imread("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/Impostor/image_A0024.jpg");
    %     queryImage = imbinarize(queryImage);
    %     queryImageFeature = extractHOGFeatures(queryImage,'CellSize',cellSize);
    %     qVector = projectData(queryImageFeature, coeff);
    
    %[label, score] = predict(myFRModel, qVector);
    % Mahalanobis distance in predicted label space.
    S = cov(subSpaceOfPredictedLabel);
    mu = mean(subSpaceOfPredictedLabel,1);
%     d = sqrt((qVector-mu)*inv(S)*(qVector-mu)');
    d = sqrt((qVector-mu)/(S)*(qVector-mu)');
    distances = [distances; d];
    
    fprintf('Distance: %.2f\n', d(1));
    %     maxDis = 0;
    %     for i = 1:length(imdsPredictedLabel.Files)
    %         queryImage = imread(imdsPredictedLabel.Files{i});
    %
    %
    %         imPostor = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/Impostor/";
    %         queryImage = imbinarize(queryImage);
    %         queryImageFeature = extractHOGFeatures(queryImage,'CellSize',cellSize);
    %         qVector = projectData(queryImageFeature, coeff);
    %
    %         % Mahalanobis distance in predicted label space.
    %         S = cov(subSpaceOfPredictedLabel);
    %         mu = mean(subSpaceOfPredictedLabel,1);
    %         d = sqrt((qVector-mu)*inv(S)*(qVector-mu)');
    %
    %         if d > maxDis
    %             maxDis = d;
    %         end
    %         % fprintf('Distance: %.2f\n', d);
    %     end
    %
    %     fprintf('Max Distance: %.2f\n', maxDis);
    %     maxDistances = [maxDistances; maxDis];
    
end

diary off