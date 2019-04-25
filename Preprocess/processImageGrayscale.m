function [processedImg] = processImageGrayscale(img, bbox, imageSize)
%This function will resize and make grayscale image of providing image 


% if input image is not grayscale, convert to grayscale
if ~isempty(bbox) == 1
    img = imcrop(img, bbox);
end

if ~isempty(imageSize) == 1
    img = imresize(img, imageSize);
end

if size(img,3)==3
    % it's RBG, we need to change to grayscale.
    
    img = rgb2gray(img);
    
end
%processedImg = histeq(img);
processedImg = adapthisteq(img);
processedImg = medfilt2(processedImg);
%imshow(processedImg);


% https://uk.mathworks.com/help/images/ref/locallapfilt.html
% Amplitude of edges, specified as a non-negative scalar. 
% sigma should be in the range [0, 1] for integer images 
% and for single images defined over the range [0, 1]. 
% For single images defined over a different range [a, b], 
% sigma should also be in the range [a, b].

% sigma = 0.125; % blur when higher
% alpha = 10; %  [0.01, 10]. alpha % Smooth
% finalProcess = locallapfilt(processedImg, sigma, alpha);
% % Amplitude of strong edges to leave intact, specified as a numeric scalar in the range [0,1].
% edgeThreshold = 0.7;
% % Amount of enhancement or smoothing desired, 
% % specified as a numeric scalar in the range [-1,1]. 
% % Negative values specify edge-aware smoothing. 
% % Positive values specify edge-aware enhancement.
% amount = 0.4;
% finalProcess = localcontrast(finalProcess, edgeThreshold, amount);

% 
%  hold on;
%  
%  
%  EyeDetect = vision.CascadeObjectDetector('EyePairBig');
%  EyeBox =step(EyeDetect,processedImg);
%  margin = 10;
%  EyeBox(1) = EyeBox(1) - margin;
%  EyeBox(2) = EyeBox(2) - margin;
%  EyeBox(3) = EyeBox(3) + margin;
%  EyeBox(4) = EyeBox(4) + margin;
%  IFaces = insertObjectAnnotation(processedImg,'rectangle',EyeBox,'eyebox');   
%  imshow(IFaces);
%  hold on;
% 
%  EyeDetect = vision.CascadeObjectDetector('LeftEye', 'UseROI', true);
%  Eye= EyeDetect(processedImg, EyeBox);
%  LeftEye  = Eye(1,:);
%  x1 = LeftEye(1) + LeftEye(2)/2;
%  y1 = LeftEye(2) + LeftEye(4)/2;
% 
%   IFaces = insertObjectAnnotation(processedImg,'rectangle',LeftEye,'LeftEye');   
%  imshow(IFaces);
%  hold on;
%  
%  plot(x1,y1, 'r*');
%  %hold on;
%  
%  EyeDetect = vision.CascadeObjectDetector('RightEye', 'UseROI', true);
%  RightEye= EyeDetect(processedImg, EyeBox);
%   IFaces = insertObjectAnnotation(processedImg,'rectangle',RightEye,'RightEye');   
%  imshow(IFaces);
%  hold on;
%  
%  for re = 1:length(RightEye)
%      rEye = RightEye(re, :);
%      if rEye(1) >= EyeBox(1) &&  rEye(1) <= (EyeBox(1) + EyeBox(3)) ...
%          && rEye(2) >= EyeBox(2) &&  rEye(2) <= (EyeBox(2) + EyeBox(4)) 
%         RightEye = rEye;
%         break;
%      end
%  end
%  
%   x2 = RightEye(1) + RightEye(2)/2;
%     y2 = RightEye(2) + RightEye(4)/2;
%  plot(x2,y2, 'g*');
% % hold on;
% 
% %  theta = atan2(y2-y1,x2-x1)*(180/pi);
% %  processedImg = imrotate(processedImg, theta);
% % 
% %  imshow(processedImg); %'bilinear', 'crop'

end

