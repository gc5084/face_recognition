clc;        % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;      % Erase all existing variables. Or clear vars.
workspace;  % Make sure the workspace panel is showing.

% Load challenge Training data
load("/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/AGC19_Challenge3_Training.mat")

% Provide the path to the input images, for example
% 'C:\AGC_Challenge_2019\images\'
imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/TRAINING/";


diary log_test_fixing_miss_detection

currentThreshold = 10;
currentMinSize = 85;
missedDetectionImages = ["image_A0024.jpg";	"image_A0056.jpg";	"image_A0063.jpg";	"image_A0066.jpg";	"image_A0075.jpg";	"image_A0095.jpg";	"image_A0100.jpg";	"image_A0133.jpg";	"image_A0144.jpg";	"image_A0145.jpg";	"image_A0147.jpg";	"image_A0162.jpg";	"image_A0197.jpg";	"image_A0198.jpg";	"image_A0205.jpg";	"image_A0211.jpg";	"image_A0212.jpg";	"image_A0225.jpg";	"image_A0227.jpg";	"image_A0240.jpg";	"image_A0251.jpg";	"image_A0253.jpg";	"image_A0258.jpg";	"image_A0263.jpg";	"image_A0276.jpg";	"image_A0283.jpg";	"image_A0285.jpg";	"image_A0299.jpg";	"image_A0305.jpg";	"image_A0312.jpg";	"image_A0348.jpg";	"image_A0351.jpg";	"image_A0368.jpg";	"image_A0384.jpg";	"image_A0386.jpg";	"image_A0399.jpg";	"image_A0413.jpg";	"image_A0431.jpg";	"image_A0473.jpg";	"image_A0488.jpg";	"image_A0519.jpg";	"image_A0527.jpg";	"image_A0540.jpg";	"image_A0560.jpg";	"image_A0581.jpg";	"image_A0596.jpg";	"image_A0598.jpg";	"image_A0617.jpg";	"image_A0630.jpg";	"image_A0635.jpg";	"image_A0638.jpg";	"image_A0646.jpg";	"image_A0652.jpg";	"image_A0658.jpg";	"image_A0688.jpg";	"image_A0694.jpg";	"image_A0724.jpg";	"image_A0731.jpg";	"image_A0759.jpg";	"image_A0762.jpg";	"image_A0814.jpg";	"image_A0908.jpg";	"image_A0931.jpg";	"image_A0954.jpg";	"image_A0964.jpg";	"image_A0967.jpg";	"image_A0983.jpg";	"image_A0988.jpg";	"image_A0995.jpg";	"image_A1009.jpg";	"image_A1026.jpg";	"image_A1033.jpg";	"image_A1042.jpg";	"image_A1046.jpg";	"image_A1053.jpg";	"image_A1070.jpg";	"image_A1073.jpg";	"image_A1087.jpg";	"image_A1090.jpg";	"image_A1099.jpg";	"image_A1112.jpg";	"image_A1134.jpg";	"image_A1135.jpg";	"image_A1141.jpg";	"image_A1154.jpg";	"image_A1187.jpg"];
for j = 1 : length( missedDetectionImages )
    A = imread( sprintf('%s%s',...
        imgPath, missedDetectionImages(j)));
    imshow(A);
    detector1 = vision.CascadeObjectDetector('UpperBody', 'MergeThreshold', currentThreshold ...
             ,'MinSize', [currentMinSize, currentMinSize]); 
    bbox1 = detector1(A);
    IFaces = insertObjectAnnotation(A,'rectangle',bbox1,'UpperBody');   
    figure;
    imshow(IFaces);
    detector2 = vision.CascadeObjectDetector('FrontalFaceLBP', 'MergeThreshold', currentThreshold ...
             ,'MinSize', [currentMinSize, currentMinSize]); 
    bbox2 = detector2(A);
    IFaces = insertObjectAnnotation(A,'rectangle',bbox2,'FrontalFaceLBP');   
    figure;
    imshow(IFaces);
    
    detector3 = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', currentThreshold); 
    bbox3 = detector3(A);
    IFaces = insertObjectAnnotation(A,'rectangle',bbox3,'Mouth');   
    figure;
    imshow(IFaces);
    
    detector4 = vision.CascadeObjectDetector('Nose', 'MergeThreshold', currentThreshold); 
    bbox4 = detector4(A);
    IFaces = insertObjectAnnotation(A,'rectangle',bbox4,'Nose');   
    figure;
    imshow(IFaces);
    
         
end


diary off