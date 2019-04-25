function plotSubplotTrainingData(trainingImages,trainingLabels, colorMode, imgWidth, imgHeight)
nImages = size(trainingImages,1);
subplotIdx = 1;
figure();
for n = 1:nImages
    % plot every 100 images
    img_text = [num2str(trainingLabels(n))];
     if colorMode == 3 
        img = trainingImages(n,:,:,:);
        img =  uint8(reshape(img,[imgWidth imgHeight 3]));
     else
         img = trainingImages(n,:,:);
         img =  uint8(reshape(img,[imgWidth imgHeight]));
     end
    %im = uint8(reshape(img,[resizeImgWidth resizeImgHeight]));
    % resizedImg = imresize(img, [resizeImgWidth resizeImgHeight]);
    plotimg = insertObjectAnnotation(img,'rectangle',[0, imgHeight - 10, imgWidth, 10],img_text, 'FontSize',50);
    subplottight(10,10, subplotIdx);
    subplotIdx = subplotIdx + 1;
    
    if colorMode == 3
        imshow(squeeze(plotimg),'border', 'tight');
    else
        imshow(squeeze(plotimg), [],'border', 'tight');
    end
    
    
    if mod(n, 100) == 0
        figure();
        subplotIdx = 1;
    end
end
end

