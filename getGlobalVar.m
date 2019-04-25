function [imageSize, hogCellSize, pcaCoeff, pcaNumComponents] = getGlobalVar()

% PCA
global globalPcaCoeff;
pcaCoeff = globalPcaCoeff;

global globalImageSize;
imageSize = globalImageSize;

global globalCellSize;
hogCellSize = globalCellSize;

global globalPcaNumComponents;
pcaNumComponents = globalPcaNumComponents;

end