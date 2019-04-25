function [accPercent] = calcAccuracy(testLabels,predictedLabels)
% -------- Accuracy Score ------%
% Get the known labels
lenSample = length(testLabels);
cntCorrect = 0;
for i = 1:lenSample
    if testLabels(i) == predictedLabels(i)
        cntCorrect = cntCorrect + 1;
    end
end
accPercent = (cntCorrect/lenSample)*100;
%fprintf('Accuracy %.2f percent', accPercent);
end

