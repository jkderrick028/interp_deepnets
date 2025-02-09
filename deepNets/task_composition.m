%% add path
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsa');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsa\util');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsa\rdm');
addpath('D:\PP\Brain_Vision\deepNets\task_composition');

%% load brain RDMs
load('D:\PP\Brain_Vision\deepNets\RDMs_4Jinkang.mat');
brain_RDMs = RDMs_subjAvg_region;

%% alexnet
%% load alexnet RDMs
RDMs_alexnet = load('D:\PP\Brain_Vision\deepNets\RDMs_alexnet.mat');
RDMs_alexnet = RDMs_alexnet.cnnRDM;

%% Convert alexnet RDMs into squares for augmenting with duplicates
for layerIndex=1:numel(RDMs_alexnet)
    RDMs_alexnet_square(layerIndex).RDM = squareRDM(RDMs_alexnet(layerIndex).RDM);
    RDMs_alexnet_square(layerIndex).name = RDMs_alexnet(layerIndex).name;
end

%% Augment RDMs_alexnet_square with duplicates
for layerIndex=1:numel(RDMs_alexnet)
   RDMs_alexnet_square(layerIndex).RDM = repmat(RDMs_alexnet_square(layerIndex).RDM, 4,4); 
end

%% Convert RDMs_alexnet to cell
for layerIndex=1:numel(RDMs_alexnet_square)
   RDMs_alexnet_cell{layerIndex}=RDMs_alexnet_square(layerIndex); 
end

%% Compose all the layers of alexnet
RDMs_alexnet_composed.RDM = zeros(size(RDMs_alexnet_cell{1}.RDM));
RDMs_alexnet_composed.name = 'alexnet_composed';

for layerIndex=1:numel(RDMs_alexnet_cell)
   RDMs_alexnet_composed.RDM = RDMs_alexnet_composed.RDM + RDMs_alexnet_cell{layerIndex}.RDM; 
end


%% vgg16
%% load vgg16 RDMs
RDMs_vgg16 = load('D:\PP\Brain_Vision\deepNets\RDMs_vgg16.mat');
RDMs_vgg16 = RDMs_vgg16.cnnRDM;

%% Convert vgg16 RDMs into squares for augmenting with duplicates
for layerIndex=1:numel(RDMs_vgg16)
    RDMs_vgg16_square(layerIndex).RDM = squareRDM(RDMs_vgg16(layerIndex).RDM);
    RDMs_vgg16_square(layerIndex).name = RDMs_vgg16(layerIndex).name;
end

%% Augment RDMs_vgg16_square with duplicates
for layerIndex=1:numel(RDMs_vgg16)
   RDMs_vgg16_square(layerIndex).RDM = repmat(RDMs_vgg16_square(layerIndex).RDM, 4,4); 
end

%% Convert RDMs_vgg16 to cell
for layerIndex=1:numel(RDMs_vgg16_square)
   RDMs_vgg16_cell{layerIndex}=RDMs_vgg16_square(layerIndex); 
end

%% Compose all the layers of vgg16
RDMs_vgg16_composed.RDM = zeros(size(RDMs_vgg16_cell{1}.RDM));
RDMs_vgg16_composed.name = 'vgg16_composed';

for layerIndex=1:numel(RDMs_vgg16_cell)
   RDMs_vgg16_composed.RDM = RDMs_vgg16_composed.RDM + RDMs_vgg16_cell{layerIndex}.RDM; 
end

%% resnet50
%% load resnet50 RDMs
RDMs_resnet50 = load('D:\PP\Brain_Vision\deepNets\RDMs_resnet50.mat');
RDMs_resnet50 = RDMs_resnet50.cnnRDM;

%% Convert resnet50 RDMs into squares for augmenting with duplicates
for layerIndex=1:numel(RDMs_resnet50)
    RDMs_resnet50_square(layerIndex).RDM = squareRDM(RDMs_resnet50(layerIndex).RDM);
    RDMs_resnet50_square(layerIndex).name = RDMs_resnet50(layerIndex).name;
end

%% Augment RDMs_resnet50_square with duplicates
for layerIndex=1:numel(RDMs_resnet50)
   RDMs_resnet50_square(layerIndex).RDM = repmat(RDMs_resnet50_square(layerIndex).RDM, 4,4); 
end

%% Convert RDMs_resnet50 to cell
for layerIndex=1:numel(RDMs_resnet50_square)
   RDMs_resnet50_cell{layerIndex}=RDMs_resnet50_square(layerIndex); 
end

%% Compose all the layers of resnet50
RDMs_resnet50_composed.RDM = zeros(size(RDMs_resnet50_cell{1}.RDM));
RDMs_resnet50_composed.name = 'resnet50_composed';

for layerIndex=1:numel(RDMs_resnet50_cell)
   RDMs_resnet50_composed.RDM = RDMs_resnet50_composed.RDM + RDMs_resnet50_cell{layerIndex}.RDM; 
end

%% Convert RDMs_deepnets_composed into cell type
RDMs_deepnets_composed_cell{1} = RDMs_alexnet_composed
RDMs_deepnets_composed_cell{2} = RDMs_vgg16_composed
RDMs_deepnets_composed_cell{3} = RDMs_resnet50_composed

%% userOptions setting
userOptions.RDMcorrelationType='Spearman';
userOptions.RDMrelatednessTest = 'subjectRFXsignedRank';
userOptions.nRandomisations = 3;
userOptions.saveFigureFig = true;
userOptions.nBootstrap = 3;
userOptions.resultsPath = 'D:\PP\Brain_Vision\deepNets\task_composition';

%% Compute correlation between brain RDMs and Deepnets Composed RDMs
stats_p_r=compareRefRDM2candRDMs(RDMs_subj_region(:,:,:,7), RDMs_deepnets_composed_cell, userOptions);