function getNetActivations(imagePath,netStr,filter)
% INPUT
% netStr    options: 'alexnet' 'vgg16' 'resnet50'


%% preparation
if ~exist('imagePath','var'), imagePath = 'D:\PP\Brain_Vision\stimuli'; end
if ~exist('netStr','var'), netStr = 'alexnet'; end
%if ~exist('filter','var'), filter = '*0.bmp'; end

if ~exist('filter','var'), filter = '*.bmp'; end

if strcmp(netStr,'alexnet'), layerIs = [2 6 10 12 14 17 20 23]; 
elseif strcmp(netStr,'vgg16'), layerIs = [2 4 7 9 12 14 16 19 21 23 26 28 30 33 36 39]; 
elseif strcmp(netStr,'resnet50'), layerIs = [2 6 9 12 13 18 21 24 28 31 34 38 41 44 45 50 53 56 60 63 66 70 73 76 80 83 86 87 92 95 98 102 105 108 112 115 118 122 125 128 132 135 138 142 145 148 149 154 157 160 164 167 170 175]; 
end


resultsPath = fullfile(imagePath,netStr); 
if ~exist(resultsPath,'dir'); mkdir(resultsPath); end

addpath('D:\PP\Brain_Vision\Jinkang_matlabCode'); % for function get_files
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsatoolbox-develop\Engines'); % if you want to show the RDMs
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsatoolbox-develop\rsa\fig');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsatoolbox-develop\rsa\rdm');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsatoolbox-develop\rsa\util');


%% control variables
monitor = 1;
distMeasure = 'correlation';


%% load network
if strcmp(netStr,'alexnet'), net = alexnet;
elseif strcmp(netStr,'vgg16'), net = vgg16;     
elseif strcmp(netStr,'resnet50'), net = resnet50; 
end
net.Layers % print layer info
inputSize = net.Layers(1).InputSize;


%% load & prepare images
% load images
files = get_files(imagePath,filter);
nImages = size(files,1);
images = nan(inputSize(1),inputSize(2),inputSize(3),nImages); % assume images are RGB
images = single(images);
for imageI = 1:nImages
    im = imread(files(imageI,:));
    if monitor, figure(10); imshow(im); title('original'); end
    % crop image if needed 
    if size(im,1) ~= size(im,2)
        im_cropped = imcrop(im); % crop manually
        if size(im_cropped,1) ~= size(im_cropped,2)
            nPix = min(size(im_cropped,1),size(im_cropped,2));
            im_cropped = im_cropped(1:nPix,1:nPix,:); % ensure image is square
        end
        if monitor, figure(11); imshow(im_cropped); title('cropped'); end
    else
        im_cropped = im;
    end
    % resize image if needed
    if size(im_cropped,1) ~= inputSize(1) || size(im_cropped,2) ~= inputSize(2)
        im_cropped_resized = imresize(im_cropped,[inputSize(1) inputSize(2)]);
        if monitor, figure(12); imshow(im_cropped_resized); title('resized'); end
    else
        im_cropped_resized = im_cropped;
    end
    % zerocenter image
    im_cropped_resized_cntrd = activations(net,im_cropped_resized,net.Layers(1).Name); % assuming first layer performs 'zerocenter' normalisation
    if monitor, figure(13); imshow(uint8(im_cropped_resized_cntrd)); title('after zerocenter normalisation'); end
    % store image
    images(:,:,:,imageI) = single(im_cropped_resized_cntrd);
end % imageI

% save
save(fullfile(resultsPath,'images'),'images');


%% compute activations
nLayers = numel(layerIs);
nUnits = nan(nLayers,1);
fileID = fopen(fullfile(resultsPath,'top5.txt'),'w');
for imageI = 1:nImages
    im = images(:,:,:,imageI);    
    acti = cell(nLayers,1);
    for layerIsI = 1:nLayers
        layerI = layerIs(layerIsI);
        layerName = net.Layers(layerI).Name;
        acti{layerIsI} = activations(net,im,layerName);
        nUnits(layerIsI) = numel(acti{layerIsI});
    end % layerI
        
    [label,scores] = classify(net,im);
    [~,idx] = sort(scores,'descend');
    idx = idx(1:5);
    top5 = net.Layers(end).ClassNames(idx);
    [pth,fle,ext] = fileparts(files(imageI,:));
    save(fullfile(resultsPath,fle),'im','acti','scores','top5');
    % print top5 to text file
    formatSpec1 = '%s\n'; formatSpec2 = '%s\n\n';
    fprintf(fileID,formatSpec1,[fle,ext]); 
    for labelI = 1:numel(idx) 
        if labelI < numel(idx), fprintf(fileID,formatSpec1,top5{labelI}); 
        else fprintf(fileID,formatSpec2,top5{labelI}); end 
    end
end % imageI
save(fullfile(resultsPath,'vars'),'nImages','nLayers','nUnits','files');
fclose(fileID);


%% compute RDMs
for layerIsI = 1:nLayers    
    layerI = layerIs(layerIsI);
    
    % original
    acti__images_units = single(nan(nImages,nUnits(layerIsI)));
     
    for imageI = 1:nImages
        [pth,fle,ext] = fileparts(files(imageI,:));
        load(fullfile(resultsPath,fle),'acti');
        acti__images_units(imageI,:) = acti{layerIsI}(:);
    end % imageI
    cnnRDM(layerIsI).RDM = pdist(acti__images_units,distMeasure);
    cnnRDM(layerIsI).name = net.Layers(layerI).Name;    
end % layerI
save(fullfile(resultsPath,['RDMs_',distMeasure]),'cnnRDM');

% display RDMs
PSfilespec = fullfile(resultsPath,'RDMs.ps');
description(1) = {['\fontsize{14}',netStr]};
description(2) = {['\fontsize{11}',distMeasure,' distance']};
figI = 500; pageFigure(figI); %showRDMs(cnnRDM,figI);
addHeadingAndPrint(description,PSfilespec,figI);
figI = 501; pageFigure(figI);

%showRDMs(cnnRDM,figI,0);

showRDMs(cnnRDM,figI);

%addHeadingAndPrint(description,PSfilespec,figI);
addHeading(description,figI,10,10);

