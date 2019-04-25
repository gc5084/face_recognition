function [bbox] = MyFaceDetectionFunction(detector, img)
% MyFaceDetectionFunction detects image for human face.
% Input Argument
% - img: input image
% Output
% - bboxes: return emtpy if face is not detected otherwise return bounding
% box, in case multiple bounding boxes have been detected, the biggest box
% area will be return.

% Detect face boundary box.
% detector = getGlobalDetector;

% if size(img,3) == 3
%      img = rgb2gray(img);
% end

bboxes = detector(img);
bbox = [];
if ~isempty(bboxes) == 1
    nFaces = size(bboxes, 1);
    if nFaces > 1
        
    % Processing bounding boxes to get the biggest one.
    
    % for keeping the largest box
    tempBoxArea = 0; 
    
    % for keeping the index of largest box
    biggestImageIdx = 0; 
    
    % Loop through all detected box
    for nFace = 1: nFaces
        % Calculate area of square
        area = bboxes(nFace, 3)* bboxes(nFace, 4);
        if area > tempBoxArea
            biggestImageIdx = nFace;
            tempBoxArea = area;
        end
    end
        bbox = bboxes(biggestImageIdx, :);
    else 
        bbox = bboxes;
    end
%     % If face is detected, show image.
%     img = insertObjectAnnotation(img,'rectangle',bbox,'Face');
%     
%     detector2 = vision.CascadeObjectDetector('ClassificationModel','Mouth', 'UseROI', true, 'MergeThreshold', 100);
%     mouthBox = detector2(img, bbox);
%     
%     if ~isempty(mouthBox) == 1
%         img = insertObjectAnnotation(img,'rectangle',mouthBox,'mouth');
%     end
%     
%     detector3 = vision.CascadeObjectDetector('ClassificationModel','EyePairSmall', 'UseROI', true, 'MergeThreshold', 10);
%     box3 = detector3(img, bbox);
%     if ~isempty(box3) == 1
%         img = insertObjectAnnotation(img,'rectangle',box3,'EyePairBig');
%     end
%     figure;
%     imshow(img);
    
    
%else
%    imwrite(img,"DetectionFailed" + num2str(incrementNumber) + ".jpg",'jpg');
end

end

