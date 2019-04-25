function [dataReprojected] = reprojectData(projectedData, eigenvectors, meanProjection)
dataReprojected = (projectedData * eigenvectors.') + meanProjection;
end

