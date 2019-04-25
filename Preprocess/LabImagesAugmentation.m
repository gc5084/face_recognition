function ImagesAugmentation()
close all;  % Close all figures (except those of imtool.)







for foldId = 1:80
    
    % !!! Change configruation here.
    % 0: No need detector to find box if images are from lab
    % 1: Use detector for bounding box
    isFaceDetectorUsed = 1; 
    % 0: When augment grayscale, otherwise for color image augmentation to
    % grayscale
    isPreprocessImageUsed = 1;
    imageSize = [128 128];
    
    %sourceImageRootFolder = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Preprocess/NewTrainImages/Orignial/" + num2str(foldId);
    sourceImageRootFolder = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/TRAINING/" + num2str(foldId);
    outputImageRootFolder= "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/LABGray128Augmented/";
    
    % Take images for each folder named by the labe_id
    imds = imageDatastore(sourceImageRootFolder, ...
        'IncludeSubfolders',true,'FileExtensions','.jpg', ...
        "LabelSource", "foldernames");
    
    exportFileType = "jpg";
    
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
        writeImageToFilePathName = outputImageRootFolder + strLabelId + "/" + imageName;
        % writeImageToFilePath = writeImageToFilePathName + "." + exportFileType;
        
        
        newSubFolder = outputImageRootFolder + strLabelId;
        if ~exist((newSubFolder), 'dir')
            mkdir(newSubFolder);
        end
        
        fprintf("processing image: %0.0f\n", j);
        
        % defect face from image by our detector model.
        if isFaceDetectorUsed ~= 0
            bbox = MyFaceDetectionFunction(A);
        else
            bbox = [0 0 size(A)];
        end
        
        % If number of face is greater than zero (detected), do the processing
        if ~isempty(bbox) == 1
            
            % write image with original bbox
            if isPreprocessImageUsed ~= 0
                processedImg = processImageGrayscale(A, bbox, imageSize);
            else
                processedImg = A;
            end
            imwrite(processedImg, ...
                writeImageToFilePathName + 'ori.' + exportFileType, exportFileType);
            %         figure;
            %         imshow(insertObjectAnnotation(A,'rectangle',bbox,'Face'));
            
            
            yTranslations = [20];
            for transIdx = 1:length(yTranslations)
                transY = yTranslations(transIdx);
                newBox = bbox;
                % y position
                newBox(2) =  bbox(2) + transY;
                %augImg = imresize(imcrop(A, newBox), imageSize);
                if isPreprocessImageUsed ~= 0
                    processedImg = processImageGrayscale(A, newBox, imageSize);
                else
                    processedImg = imresize(imcrop(A, newBox), imageSize);
                end
                imwrite(processedImg, ...
                    writeImageToFilePathName + 'yTran' + num2str(transY) + '.' + exportFileType, exportFileType);
                %             figure;
                %             imshow(insertObjectAnnotation(A,'rectangle',newBox,'YTranslation'));
            end
            
            
            rotateDegree = [-10 10];
            for rotateIdx = 1:length(rotateDegree)
                angle = rotateDegree(rotateIdx);
                
                newBox = bbox;
                % x position
                newBox(1) =  bbox(1) + 20;
                % y position
                newBox(2) =  bbox(2) + 20;
                augImg = imresize(imcrop(imrotate(A,angle), newBox), imageSize);
                if isPreprocessImageUsed ~= 0
                    processedImg = processImageGrayscale(augImg, [], []);
                else
                    processedImg = augImg;
                end
                imwrite(processedImg, ...
                    writeImageToFilePathName + 'angle_' + num2str(angle) + '.' + exportFileType, exportFileType);
                %             figure;
                %             imshow(insertObjectAnnotation(imrotate(A,angle),'rectangle',newBox,'rotate'));
            end
            
            % zoom in
            zoomPixSize = [10];
            for scaleSize = 1:length(zoomPixSize)
                pixSize = zoomPixSize(scaleSize);
                
                newBox = zeros(1, 4);
                % x position
                newBox(1) =  bbox(1) + pixSize;
                % y position
                newBox(2) =  bbox(2) + pixSize;
                % width
                newBox(3) =  bbox(3) - pixSize*2;
                % height
                newBox(4) =  bbox(4) - pixSize*2;
                augImg = imresize(imcrop(A, newBox), imageSize);
                if isPreprocessImageUsed ~= 0
                    processedImg = processImageGrayscale(augImg, [], []);
                else
                    processedImg = augImg;
                end
                imwrite(processedImg, ...
                    writeImageToFilePathName + 'zoomin_' + num2str(pixSize) + '.' + exportFileType, exportFileType);
                %             figure;
                %             imshow(insertObjectAnnotation(A,'rectangle',newBox,'zoom in'));
            end
            
            % scale: positive number less than 1 or greater
            % scalesPixSize = [10 20 30];
            % zoom out
            zoomoutPixSize = [25];
            for scaleSize = 1:length(zoomoutPixSize)
                pixSize = zoomoutPixSize(scaleSize);
                
                newBox = zeros(1, 4);
                % x position
                newBox(1) =  bbox(1) - pixSize;
                % y position
                newBox(2) =  bbox(2) - pixSize;
                % width
                newBox(3) =  bbox(3) + pixSize*2;
                % height
                newBox(4) =  bbox(4) + pixSize*2;
                augImg = imresize(imcrop(A, newBox), imageSize);
                if isPreprocessImageUsed ~= 0
                    processedImg = processImageGrayscale(augImg, [], []);
                else
                    processedImg = augImg;
                end
                imwrite(processedImg, ...
                    writeImageToFilePathName + 'zoomout_' + num2str(pixSize) + '.' + exportFileType, exportFileType);
                % figure;
                % imshow(insertObjectAnnotation(A,'rectangle',newBox,'zoom out'));
            end
        end
    end
end
end
