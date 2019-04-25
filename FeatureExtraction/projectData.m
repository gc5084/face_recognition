function [projectedData] = projectData(X, eigenvectors)
projectedData = X * (eigenvectors);
end

