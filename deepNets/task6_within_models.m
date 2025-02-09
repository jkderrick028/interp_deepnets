%% add path
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsatoolbox-develop\rsa');

%% load deep nets RDMs
RDMs_alexnet = load('D:\PP\Brain_Vision\deepNets\RDMs_alexnet.mat');
RDMs_alexnet = RDMs_alexnet.cnnRDM;
for modelLayerIndex=1:numel(RDMs_alexnet)
    RDMs_alexnet_cell{modelLayerIndex}=RDMs_alexnet(modelLayerIndex);
end
%% userOptions
userOptions.RDMcorrelationType='Spearman';
userOptions.RDMrelatednessTest='subjectRFXsignedRank';
%% compute pairwise correlations
stats_p_r = compareRefRDM2candRDMs(RDMs_alexnet, RDMs_alexnet_cell, userOptions);