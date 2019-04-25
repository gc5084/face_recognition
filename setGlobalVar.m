function  setGlobalVar(imageSize, hogCellSize, pcaCoeff, pcaNumComponents)

% PCA
global globalPcaCoeff;
globalPcaCoeff = pcaCoeff;

global globalImageSize;
globalImageSize = imageSize;

global globalCellSize;
globalCellSize = hogCellSize;

global globalPcaNumComponents;
globalPcaNumComponents = pcaNumComponents;

end