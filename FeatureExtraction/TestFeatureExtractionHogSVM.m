clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

diary log_hog_clean_merge_data

% imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128/ID/";
% imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Augmented/";
imgPath = "F:\spain\UPF\course\face and gesture analysis\project_4\team_code\new_code\FaceAndGesture-Lab4\Models\DS\AugmentedGrayscale128\";
% imgPath = "/Volumes/Work/UPF/Class_FACIAL/LAB2-FG/CKDB/";
imds = imageDatastore(imgPath, ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg'}, "LabelSource", "foldernames");

%[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');


% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

% Tunable parameter
cellSize = [13 13];
imageSize = [128 128];

imgTemp = imread(imds.Files{1});
[hog, vis] = extractHOGFeatures(imgTemp,'CellSize',cellSize);
hogFeatureSize = length(hog);
% numImages = numel(trainingSet.Files);


% ====
% Get labels for each image.
[trainingFeatures, trainingLabels] = helperExtractHOGFeatures(imds, hogFeatureSize, cellSize);

% cumluative sum
%[coeff,score,latent,tsquared,explained,mu] = pca(trainingFeatures,  'Centered', true);
% plotCumsumPCA(explained);


% 10 50 100 150  200 300 400 600 650 700 750 800
% CV - best cv result = 500 = 41.7
nComponents = [500];
for n=1:length(nComponents)
    
    [coeff,score,latent,tsquared,explained,mu] = pca(trainingFeatures, ...
        'NumComponents', nComponents(n), 'Centered', true);
    
    dataProjected = projectData(trainingFeatures, coeff);
    
    % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
    % https://uk.mathworks.com/help/stats/fitcecoc.html
    %     myFRModel = fitcecoc(dataProjected, trainingLabels);%, ...
    %'OptimizeHyperparameters', 'auto', 'Verbose', 1);
    classifier = fitcecoc(dataProjected, trainingLabels);

    % Extract HOG features from the test set. The procedure is similar to what
    % was shown earlier and is encapsulated as a helper function for brevity.
    %     [testFeatures, testLabels] = helperExtractHOGFeatures(testSet, ...
    %         hogFeatureSize, cellSize);
    %     testDataProjected = projectData(testFeatures, coeff);
    
    
    % Make class predictions using the test features.
    % [predictedLabels, scores] = predict(myFRModel, testDataProjected);
    
    % Calc accuracy
    %accuracy = calcAccuracy(testLabels, predictedLabels);
    %fprintf('PCA %.0f SVM: %.2f\n', nComponents(n),  accuracy);
    
    %https://uk.mathworks.com/help/stats/fitcecoc.html
    CVMdl = crossval(classifier, 'KFold', 3);
    loss = kfoldLoss(CVMdl);
    fprintf('Accuracy from CV: %.2f\n', (1 - loss)*100);
    
        
    my_FRmodel = struct('classifier', classifier, 'faceDetector', vision.CascadeObjectDetector('MergeThreshold', 5, 'MinSize', [85 85]),  'imageSize', [128 128], 'cellSize', [13 13], 'nComponents', 500, 'pcaCoeff', coeff);
    
    save 'googleImgRot.mat' 'my_FRmodel';
    
end
diary off