function   setGlobalDetector(MergeThreshold,minSize)
% Set value properties of face detector model 
% MergeThreshold is MergeThreshold properties of CascadeObjectDetector
% minSize is Minimum size properties of CascadeObjectDetector
global detector;
detector = vision.CascadeObjectDetector('MergeThreshold',MergeThreshold, 'MinSize', minSize);
%detector = vision.CascadeObjectDetector();
end