% DEMO1_RSA_ROI_simulatedAndRealData
%
% This function computes the results in the main Figures of 
% Nili et al. (PLoS Comp Biol 2013)

%% addpath
% addpath(genpath('D:\PP\Brain_Vision\RSAtoolbox'));
addpath('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData');
addpath('D:\PP\Brain_Vision\RSAtoolbox');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsa');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsa\rdm');
addpath('D:\PP\Brain_Vision\RSAtoolbox\rsa\fig');

%%
clear;clc; close all hidden;
returnHere = pwd;
cd ..;addpath(genpath(pwd));cd(returnHere)
userOptions = defineUserOptions;
mkdir('DEMO1');
userOptions.rootPath = [pwd,filesep,'DEMO1'];
userOptions.analysisName = 'DEMO1';

%% control variables
nSubjects=12;
subjectPatternNoiseStd=1;

nModelGrades=3;
bestModelPatternDeviationStd=0;
worstModelPatternDeviationStd=6;

patternDistanceMeasure='correlation';


%% load RDMs and category definitions from Kriegeskorte et al. (Neuron 2008)
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\Kriegeskorte_Neuron2008_supplementalData.mat');
rdm_mIT=squareRDMs(RDMs_mIT_hIT_fig1(1).RDM);
rdm_hIT=squareRDMs(RDMs_mIT_hIT_fig1(2).RDM);

load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\92_brainRDMs.mat')
RDMs_hIT_bySubject = averageRDMs_subjectSession(RDMs, 'session');
showRDMs(RDMs_hIT_bySubject,1);
handleCurrentFigure([userOptions.rootPath,filesep,'subjectRDMs_hIT_fMRI'],userOptions);


%% load reconstructed patterns for simulating models
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\simTruePatterns.mat','simTruePatterns','simTruePatterns2')
[nCond nDim]=size(simTruePatterns);

%% simulate multiple subjects' noisy RDMs
subjectRDMs=nan(nCond,nCond,nSubjects);

for subjectI=1:nSubjects
    patterns_cSubject=simTruePatterns2+subjectPatternNoiseStd*randn(nCond,nDim);
    subjectRDMs(:,:,subjectI)=squareRDMs(pdist(patterns_cSubject,patternDistanceMeasure));
end

avgSubjectRDM=mean(subjectRDMs,3);

showRDMs(concatRDMs_unwrapped(subjectRDMs,avgSubjectRDM),2);
handleCurrentFigure([userOptions.rootPath,filesep,'simulatedSubjAndAverage'],userOptions);


%% define categorical model RDMs
[binRDM_animacy, nCatCrossingsRDM]=categoricalRDM(categoryVectors(:,1),3,true);
ITemphasizedCategories=[1 2 5 6]; % animate, inanimate, face, body
[binRDM_cats, nCatCrossingsRDM]=categoricalRDM(categoryVectors(:,ITemphasizedCategories),4,true);
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\faceAnimateInaniClustersRDM.mat')


%% load behavioural RDM from Mur et al. (Frontiers Perc Sci 2013)
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\92_behavRDMs.mat')
rdm_simJudg=mean(stripNsquareRDMs(rdms_behav_92),3);


%% create modelRDMs of different degrees of noise
gradedModelRDMs=nan(nCond,nCond,nModelGrades);

patternDevStds=linspace(bestModelPatternDeviationStd,worstModelPatternDeviationStd,nModelGrades);

for gradedModelI=1:nModelGrades
    patterns_cGradedModel=simTruePatterns2+patternDevStds(gradedModelI)*randn(nCond,nDim);
    gradedModelRDMs(:,:,gradedModelI)=squareRDMs(pdist(patterns_cGradedModel,patternDistanceMeasure));
end


%% load RDMs for V1 model and HMAX model with natural image patches from Serre et al. (Computer Vision and Pattern Recognition 2005)
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\rdm92_V1model.mat')
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\rdm92_HMAXnatImPatch.mat')


%% load RADON and silhouette models and human early visual RDM
load('D:\PP\Brain_Vision\RSAtoolbox\Demos\92imageData\92_modelRDMs.mat');
FourCatsRDM=Models(2).RDM;
humanEarlyVisualRDM=Models(4).RDM;
silhouetteRDM=Models(7).RDM;
radonRDM=Models(8).RDM;


%% concatenate and name the modelRDMs
modelRDMs=cat(3,binRDM_animacy,faceAnimateInaniClustersRDM,FourCatsRDM,rdm_simJudg,humanEarlyVisualRDM,rdm_mIT,silhouetteRDM,rdm92_V1model,rdm92_HMAXnatImPatch,radonRDM,gradedModelRDMs);
modelRDMs=wrapAndNameRDMs(modelRDMs,{'ani./inani.','face/ani./inani.','face/body/nat./artif.','sim. judg.','human early visual','monkey IT','silhouette','V1 model','HMAX-2005 model','RADON','true model','true with noise','true with more noise'});
modelRDMs=modelRDMs(1:end-2); % leave out the true with noise models

showRDMs(modelRDMs,5);
handleCurrentFigure([userOptions.rootPath,filesep,'allModels'],userOptions);
% place the model RDMs in cells in order to pass them to
% compareRefRDM2candRDMs as candidate RDMs
for modelRDMI=1:numel(modelRDMs)
    modelRDMs_cell{modelRDMI}=modelRDMs(modelRDMI);
end

%% activity pattern MDS
categoryIs=[5 6 7 8];
categoryCols=[0 0 0
              0 0 0
              0 0 0
              0 0 0
              1 0.5 0
              1 0 0
              0 1 0
              0 0.5 1];



% MDS plot
categoryIs=[5 6 7 8];
categoryCols=[0 0 0
    0 0 0
    0 0 0
    0 0 0
    1 0.5 0
    1 0 0
    0 1 0
    0 0.5 1];


for condI = 1:92
    for catI = 1:numel(categoryIs)
        if categoryVectors(condI,categoryIs(catI))
            userOptions.conditionColours(condI,:) = categoryCols(categoryIs(catI),:);
        end
    end
end
avgRDM.RDM = avgSubjectRDM;
avgRDM.name = 'subject-averaged RDM';
avgRDM.color = [0 0 0];
[blankConditionLabels{1:size(modelRDMs_cell{1}.RDM,1)}] = deal(' ');

% true-model MDS
MDSConditions(modelRDMs_cell{11}, userOptions,struct('titleString','ground-truth MDS',...
    'fileName','trueRDM_MDS','figureNumber',6));
% true-model dendrogram

dendrogramConditions(modelRDMs_cell{11}, userOptions,...
struct('titleString', 'Dendrogram of the ground truth RDM', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 7));
% subject-averaged MDS
MDSConditions(avgRDM, userOptions,struct('titleString','subject-averaged MDS',...
    'fileName','ssMDS','figureNumber',8));
% subject-averaged Dendrogram
dendrogramConditions(avgRDM, userOptions,...
struct('titleString', 'Dendrogram of the subject-averaged RDM', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 9));

% one-subject MDS (e.g. simulated subject1), noisier
MDSConditions(rsa.rdm.wrapAndNameRDMs(subjectRDMs(:,:,1),{'single-subject RDM'}), userOptions,struct('titleString','sample subject MDS',...
    'fileName','single-subject RDM','figureNumber',10));

% one-subject Dendrogram
dendrogramConditions(wrapAndNameRDMs(subjectRDMs(:,:,3),{'single-subject RDM'}), userOptions,...
struct('titleString', 'Dendrogram of a single-subject RDM', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 11));



%% RDM correlation matrix and MDS
% 2nd order correlation matrix
userOptions.RDMcorrelationType='Kendall_taua';

pairwiseCorrelateRDMs({avgRDM, modelRDMs}, userOptions, struct('figureNumber', 12,'fileName','RDMcorrelationMatrix'));

% 2nd order MDS
MDSRDMs({avgRDM, modelRDMs}, userOptions, struct('titleString', 'MDS of different RDMs', 'figureNumber', 13,'fileName','2ndOrderMDSplot'));


%% statistical inference
userOptions.RDMcorrelationType='Kendall_taua';
userOptions.RDMrelatednessTest = 'subjectRFXsignedRank';
userOptions.RDMrelatednessThreshold = 0.05;
userOptions.RDMrelatednessMultipleTesting = 'FDR';
userOptions.saveFiguresPDF = 1;
userOptions.candRDMdifferencesTest = 'subjectRFXsignedRank';
userOptions.candRDMdifferencesThreshold = 0.05;
userOptions.candRDMdifferencesMultipleTesting = 'FDR';
userOptions.plotpValues = '=';
userOptions.barsOrderedByRDMCorr=true;
userOptions.resultsPath = userOptions.rootPath;
userOptions.figureIndex = [14 15];
userOptions.figure1filename = 'compareRefRDM2candRDMs_barGraph_simulatedITasRef';
userOptions.figure2filename = 'compareRefRDM2candRDMs_pValues_simulatedITasRef';
stats_p_r=compareRefRDM2candRDMs(subjectRDMs, modelRDMs_cell, userOptions);


%% Finally: real fMRI data (human IT RDM from Kriegeskorte et al. (Neuron 2008) as the reference RDM
% userOptions.RDMcorrelationType='Kendall_taua';
userOptions.RDMcorrelationType='Spearman';

userOptions.RDMrelatednessTest = 'randomisation';
userOptions.RDMrelatednessThreshold = 0.05;
userOptions.RDMrelatednessMultipleTesting = 'none';%'FWE'
userOptions.candRDMdifferencesTest = 'conditionRFXbootstrap';
userOptions.candRDMdifferencesMultipleTesting = 'FDR';
userOptions.plotpValues = '*';
userOptions.nRandomisations = 100;
userOptions.nBootstrap = 100;
userOptions.candRDMdifferencesThreshold = 0.05;
userOptions.candRDMdifferencesMultipleTesting = 'FDR';
userOptions.figure1filename = 'compareRefRDM2candRDMs_barGraph_hITasRef';
userOptions.figure2filename = 'compareRefRDM2candRDMs_pValues_hITasRef';
userOptions.figureIndex = [16 17];
stats_p_r=compareRefRDM2candRDMs(RDMs_hIT_bySubject, modelRDMs_cell(1:end-1), userOptions);
