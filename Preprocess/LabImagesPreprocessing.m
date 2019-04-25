clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.


% Load challenge Training data
load("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat")

% Provide the path to the input images, for example
% 'C:\AGC_Challenge_2019\images\'
imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";


% !!!!! Set color and size here
colorMode = 3; % 1: Grayscale, 3: Color RGB
imageSize = [224 224];
% !!!! Change target folder
rootWriteImagePath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABRGB224/ID/";
rootWriteImpostorImagePath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABRGB224/Impostor/";


if ~exist((rootWriteImagePath), 'dir')
        mkdir(rootWriteImagePath);
end
if ~exist((rootWriteImpostorImagePath), 'dir')
        mkdir(rootWriteImpostorImagePath);
end


trainingImages = [];
trainingLabels = [];
boxesSize = [];
imagesSize = [];

nMissDetectName = [];
subplotImages = [];



% Process all images in the Training set
for j = 1 :length( AGC19_Challenge3_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC19_Challenge3_TRAINING(j).imageName ));
    
      % Get label ID from training set
    label_id = AGC19_Challenge3_TRAINING(j).id;
    imageName = AGC19_Challenge3_TRAINING(j).imageName;
    
    writeImageToFilePath = rootWriteImagePath+num2str(label_id)+"/"+imageName;
    
    newSubFolder = rootWriteImagePath+num2str(label_id);
    if ~exist((newSubFolder), 'dir')
        mkdir(newSubFolder);
    end
    
    fprintf("processing image: %0.0f\n", j);

    bboxes = AGC19_Challenge3_TRAINING(j).faceBox;
    % Get number of detected face
    nFaces = size(bboxes, 1);
    
    if nFaces > 0
        
        for f = 1:nFaces
            % Process box size from [x1 y1 x2 y2] to [x y width height]
            bboxes(f, 3) = bboxes(f, 3) - bboxes(f, 1);
            bboxes(f, 4) = bboxes(f, 4) - bboxes(f, 2);

            if nFaces == 1
                % if there is only one face detected
                % add face image to training image and add label to
                % trainning label
                % pass image to be processed in processImage function
                if colorMode == 1
                    processedImage = processImageGrayscale(A, bboxes(f, :), ...
                        imageSize) ;
                    trainingImages(j,:,:) = processedImage;
                    trainingLabels(j,:) = label_id;
                    
                else 
                    processedImage = processImageRGB(A, bboxes(f, :), ...
                        imageSize) ;
                    trainingImages(j,:,:,:) = processedImage;
                    trainingLabels(j,:) = label_id;
                end
                
                imwrite(processedImage,writeImageToFilePath,'jpg');
                % boxesSize is for calculating the mean of width and height
                % of the image for resizing
                boxesSize = [boxesSize; bboxes];
            elseif nFaces > 1
                fprintf("image name: %s", AGC19_Challenge3_TRAINING(j).imageName);
                %                  IFaces = insertObjectAnnotation(A,'rectangle',bboxes,'Face');
                %                 figure;
                %                 imshow(IFaces);
                % If number of face is more than 1,
                % as from assignment we have to label the biggest one as the
                % id.
                
                % Find the biggest box's size and set label
                % from rectangle area width x height
                % which is at the index 3, 4 of bbox respectively
                
                tempBoxArea = 0; % for keeping the largest box
                biggestImage = 0; % for keeping the index of largest box
                % Loop through all detected box
                for nImg = 1: nFaces
                    % Calculate area of square
                    area = bboxes(nImg, 3)* bboxes(nImg, 4);
                    if area > tempBoxArea
                        biggestImage = nImg;
                        tempBoxArea = area;
                    end
                end
                
                % Label the biggest image with ID
                if colorMode == 1
                    processedImage = processImageGrayscale(A, bboxes(biggestImage, :) ...
                        , imageSize) ;

                    trainingImages(j,:,:) = processedImage;
                    trainingLabels(j,:) = label_id;
                else
                    processedImage = processImageRGB(A, bboxes(f, :), ...
                        imageSize) ;
                    trainingImages(j,:,:,:) = processedImage;
                    trainingLabels(j,:) = label_id;
                end
                boxesSize = [boxesSize; bboxes(biggestImage, :)];
                imwrite(processedImage,writeImageToFilePath,'jpg');
            end
        end
    else
        % -1 Label for non-face and impostor
        % Use our detector to get impostor
        impostorBBox = MyFaceDetectionFunction(A);
        if isempty(impostorBBox) ~= 1
            if colorMode == 1
                processedImage = processImageGrayscale(A, impostorBBox, ...
                    imageSize) ;
                trainingImages(j,:,:) = processedImage;
            else
                 processedImage = processImageRGB(A, impostorBBox, ...
                        imageSize) ;
                 trainingImages(j,:,:,:) = processedImage;
            end
            trainingLabels(j,:) = label_id;
            imwrite(processedImage,rootWriteImpostorImagePath+imageName,'jpg');
        else
            if colorMode == 1
                processedImage = processImageGrayscale(A, ...
                    [0 0 size(A,1) size(A,2)], imageSize) ;
                trainingImages(j,:,:) = processedImage;
            else
                 processedImage = processImageRGB(A, ...
                    [0 0 size(A,1) size(A,2)], imageSize) ;
                 trainingImages(j,:,:,:) = processedImage;
            end
            trainingLabels(j,:) = label_id;
            imwrite(processedImage,writeImageToFilePath,'jpg');
        end
    end

end

fprintf("size of images in array:%0.0f\n", size(trainingImages,1));
fprintf("size of label in array:%0.0f\n", size(trainingLabels,1));


% ==== Plotting training images as subplot
% == Uncomment if you want to see subplot of training images and labels.
% plotSubplotTrainingData(trainingImages, trainingLabels, ...
%     meanImageWidth, meanImageHeight);
