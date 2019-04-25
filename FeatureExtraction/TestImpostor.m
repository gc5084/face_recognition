% function [identityId] = TestImpostor(faceRecoguntionModel, image)

function  TestImpostor()

% if ~isempty(faceRecoguntionModel) == 1
%     modelFile = 'myFaceRecognitionModelClearnMergedGrayscale128.mat';
% load(modelFile);
% faceRecoguntionModel =  myFRModel;
% 
% end

modelFile = 'myFaceRecognitionModelClearnMergedGrayscale128.mat';
load(modelFile);
faceRecoguntionModel =  my_FRModel;


eu_distances = 'euclidean_distances.mat';
load(eu_distances);


% root = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";
root = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/Validation/";

images = ["1/1.jpg" "21/1.jpg"];
% 
% images = ["image_A0014.jpg";	'image_A0017.jpg';	'image_A0024.jpg';	'image_A0041.jpg';	'image_A0045.jpg';	'image_A0046.jpg';	'image_A0047.jpg';	'image_A0066.jpg';	'image_A0075.jpg';	'image_A0080.jpg';	'image_A0092.jpg';	'image_A0095.jpg';	'image_A0098.jpg';	'image_A0100.jpg';	'image_A0101.jpg';	'image_A0111.jpg';	'image_A0120.jpg';	'image_A0128.jpg';	'image_A0131.jpg';	'image_A0133.jpg';	'image_A0147.jpg';	'image_A0151.jpg';	'image_A0153.jpg';	'image_A0154.jpg';	'image_A0155.jpg';	'image_A0156.jpg';	'image_A0162.jpg';	'image_A0164.jpg';	'image_A0180.jpg';	'image_A0184.jpg';	'image_A0193.jpg';	'image_A0194.jpg';	'image_A0195.jpg';	'image_A0205.jpg';	'image_A0206.jpg';	'image_A0211.jpg';	'image_A0212.jpg';	'image_A0213.jpg';	'image_A0218.jpg';	'image_A0220.jpg';	'image_A0223.jpg';	'image_A0225.jpg';	'image_A0227.jpg';	'image_A0230.jpg';	'image_A0238.jpg';	'image_A0239.jpg';	'image_A0247.jpg';	'image_A0250.jpg';	'image_A0251.jpg';	'image_A0254.jpg';	'image_A0255.jpg';	'image_A0263.jpg';	'image_A0273.jpg';	'image_A0276.jpg';	'image_A0283.jpg';	'image_A0284.jpg';	'image_A0285.jpg';	'image_A0295.jpg';	'image_A0300.jpg';	'image_A0316.jpg';	'image_A0317.jpg';	'image_A0326.jpg';	'image_A0328.jpg';	'image_A0332.jpg';	'image_A0339.jpg';	'image_A0342.jpg';	'image_A0346.jpg';];

for i = 1:length(images)
    if length(images) == 1
    else
         image = imread(root + images(i));
    end
   
    
    bbox = MyFaceDetectionFunction(faceRecoguntionModel.faceDetector, image);
    
    
    
    % !!! Tunable parameter or required settings
    % cellSize, image size
    
    % Get number of detected face
    if ~isempty(bbox) == 1
        
        %     [imageSize, hogCellSize, pcaCoeff, pcaNumComponents] = getGlobalVar();
        % Face is deteced then do following:
        % ----- preprocess images
        % ----- feature extraction (HOG, PCA)
        % ----- Predict by the model
        
        % FOR CNN NN
        % identityId = classify(faceRecoguntionModel,image);
        %YTest = imdsTest.Labels;
        
        
        % FOR SVM - CNN
        %     rgbImage = imresize(imcrop(image, bbox), faceRecoguntionModel.imageSize);
        %     if size(rgbImage,3) ~= 3
        %         rgbImage=rgbImage(:,:,[1 1 1]);
        %     end
        
        %     modelFile = 'cnnModelLab.mat';
        %     load(modelFile, 'cnnModelLab');
        %     net = vgg16;
        %     featuresTest = activations(net,rgbImage,'fc7','OutputAs','rows');
        %     impostorCheck = predict(cnnModelLab, featuresTest);
        %     if impostorCheck ~= num2str(-1)
        %     % 1. Preprocess images
        
        processedImage = processImageGrayscale(image, bbox, ...
            faceRecoguntionModel.imageSize);
        imshow(processedImage);
        % 2. Feature extraction
        dataProjected = extractFeature(processedImage, ...
            faceRecoguntionModel.cellSize, faceRecoguntionModel.pcaCoeff);
        
        % 3. Predict by the model
        % !!! Change to Jess function here.
        [label, score] = predict(faceRecoguntionModel.classifier, dataProjected);
        
        labelLen = length(label);
        convertedPredictedLabel = [];
        for c = 1:labelLen
            convertedPredictedLabel = [convertedPredictedLabel,char(label(c))];
        end
        
        %identityId = str2double(convertedLabel);
        % ============ Thresholding Start =================
        %     imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/AugmentedGrayscale128/" + convertedLabel + "/";
        %
        %     imdsPredictedLabel = imageDatastore(imgPath, ...
        %         'IncludeSubfolders',true,'FileExtensions',{'.jpg'});
        
        
        [feature, visualization] = extractHOGFeatures(imbinarize(processedImage),...
            'CellSize',faceRecoguntionModel.cellSize);
        
        subSpaceOfPredictedLabel = projectData(feature, faceRecoguntionModel.pcaCoeff);
        
        S = cov(subSpaceOfPredictedLabel);
        mu = mean(subSpaceOfPredictedLabel,1);
        
        % d = sqrt((dataProjected-mu)*inv(S)*(dataProjected-mu)');
        
        % get max distance
        maxDistance = group_distances(str2num(convertedPredictedLabel), 3);
        observeDistance = pdist([group_distances(str2num(convertedPredictedLabel),4:503); dataProjected],'euclidean');
        
        if observeDistance > maxDistance
            identityId = -1;
            fprintf('predicted: %s max: %.2f\t impostor_distance: %.2f\n', ...
            convertedPredictedLabel, maxDistance, observeDistance);
        else
            identityId = str2double(convertedPredictedLabel);
        end
        % ============ Thresholding End =================
        
        
    else
        identityId = -1;
    end
end