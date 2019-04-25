function [trainingImages, trainingLabels, boxesSize] = PreprocessingImages(sourceImageRootFolder, outputImageRootFolder, outputFailedImageRootFolder, imageMode, isCalcWidthHeight, imageSizes)
% Take images for each folder named by the labe_id
imds = imageDatastore(sourceImageRootFolder, ...
    'IncludeSubfolders',true,'FileExtensions','.jpg', ...
    "LabelSource", "foldernames");

trainingImages = [];
trainingLabels = [];
boxesSize = [];
exportFileSize = "jpg"; %.jpg

% Default AlexNet take 227x227x3
% Default VGG16 takes 224x224x3
meanImageWidth = imageSizes(1);
meanImageHeight = imageSizes(2);

% Our mean width and height for bounding boxes.
% meanImgWidth = 240 %240.9694;
% meanImgHeight = 231%231.3629;
numberOfImages = size(imds.Files);
% Process all images in the Training set
incrementNumber = 0;
for j = 1:numberOfImages
    incrementNumber = incrementNumber + 1;
    
    % Read images from original folder
    A  = readimage(imds,j);
    
    % Get label ID from training set
    label_id = imds.Labels(j);
    strLabelId = cellstr(label_id);
    % Prepare new image name to write as output image
    imageName = strLabelId + "_" + num2str(incrementNumber) ;

    
    % Write image file, if the folder does not exist, create folder by
    % label_id
    writeImageToFilePath = outputImageRootFolder + strLabelId + "/" + imageName + "." + exportFileSize;
    newSubFolder = outputImageRootFolder + strLabelId;
    if ~exist((newSubFolder), 'dir')
        mkdir(newSubFolder);
    end
    
    fprintf("processing image: %0.0f\n", j);
    
    % defect face from image by our detector model.
    [bboxes] = MyFaceDetectionFunction(A);
    
    % Get number of detected face
    nFaces = size(bboxes, 1);
    
    % If number of face is greater than zero (detected), do the processing
    if nFaces > 0
        
        for f = 1:nFaces
            
            if isCalcWidthHeight == 1
                % Process box size from [x1 y1 x2 y2] to [x y width height]
                % when data provided from professor that support 
                bboxes(f, 3) = bboxes(f, 3) - bboxes(f, 1);
                bboxes(f, 4) = bboxes(f, 4) - bboxes(f, 2);
            end
            
            if nFaces == 1
                % if there is only one face detected
                % add face image to training image and add label to
                % trainning label
                % pass image to be processed in processImage function
                if imageMode == 1
                    % Grayscale
                    processedImage = processImageGrayscale(A, bboxes(f, :), ...
                        imageSizes) ;
                    trainingImages(j,:,:) = processedImage;
                    
                else
                    % RGB
                    processedImage = processImageRGB(A, bboxes(f, :), ...
                        meanImageWidth, meanImageHeight) ;
                    trainingImages(j,:,:,:) = processedImage;
                    
                end
                trainingLabels(j,:) = label_id;
                imwrite(processedImage,writeImageToFilePath, exportFileSize);
                
                % boxesSize is for calculating the mean of width and height
                % of the image for resizing
                boxesSize = [boxesSize; bboxes];
            elseif nFaces > 1
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
                
                if imageMode == 1
                    % Grayscale
                    processedImage = processImageGrayscale(A, bboxes(f, :), ...
                        imageSizes) ;
                    trainingImages(j,:,:) = processedImage;
                    
                else
                    % RGB
                    processedImage = processImageRGB(A, bboxes(f, :), ...
                        meanImageWidth, meanImageHeight) ;
                    trainingImages(j,:,:,:) = processedImage;
                    
                end
                trainingLabels(j,:) = label_id;
                imwrite(processedImage,writeImageToFilePath, exportFileSize);
                boxesSize = [boxesSize; bboxes(biggestImage, :)];
                
            end
        end
    else
        % No face detected in image
        
%         if imageMode == 1
%             % Grayscale
%             processedImage = processImageGrayscale(A, [0 0 size(A,1) size(A,2)], ...
%             imageSizes) ;
%             trainingImages(j,:,:) = processedImage;
%             
%         else
%             % RGB
%             processedImage = processImageRGB(A, [0 0 size(A,1) size(A,2)], ...
%                 meanImageWidth, meanImageHeight) ;
%             trainingImages(j,:,:,:) = processedImage;
%             
%         end
        % Write output to failed folder
        imwrite(A,outputFailedImageRootFolder + num2str(incrementNumber) + "." + exportFileSize, exportFileSize);
        boxesSize = [boxesSize; bboxes];
        
        
        
    end
    
end

fprintf("size of images in array:%0.0f\n", size(trainingImages,1));
fprintf("size of label in array:%0.0f\n", size(trainingLabels,1));


% ==== Plotting training images as subplot
% == Uncomment if you want to see subplot of training images and labels.
% plotSubplotTrainingData(trainingImages, trainingLabels, ...
%     meanImageWidth, meanImageHeight);



% === code may use
%                  IFaces = insertObjectAnnotation(A,'rectangle',bboxes,'Face');
%                 figure;
%                 imshow(IFaces);
end
