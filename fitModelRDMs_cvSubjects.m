function fitModelRDMs_cvSubjects(options)



%% preparation
dataPath = 'D:\projects\flexibleCoding\analysis\fMRI\results\pattern_information\ROI\RSA\glm_SD_concat_stimulus&task_native\ldt\cv1\avgdAcrossHemispheres'; 
modelPath = 'D:\projects\flexibleCoding\stimuli\deepNets';
savePath = fullfile(dataPath,'deepNets'); try mkdir(savePath); end

addpath(genpath('D:\projects\flexibleCoding\analysis\fMRI\rsa_local\RSAtoolbox\rsatoolbox'));

if ~exist('options','var') || isempty(options)
  options = struct();
end
if ~isfield(options,'netString')
    options.netString = 'alexnet';
end
   

%% load data and model RDMs
load(fullfile(dataPath,'RDMs'),'RDMs__subj_region_size');
dataRDMs = RDMs__subj_region_size(:,:,:,:,3); % select 71 voxels
[nConds,nConds,nSubjects,nRegions] = size(dataRDMs);
for regionI = 1:nRegions
    for subjectI = 1:nSubjects
        dataRDMs_vec(:,subjectI,regionI) = vectorizeRDM(dataRDMs(:,:,subjectI,regionI));
    end % subjectI
end % regionI
dataRDMs_vec = dataRDMs_vec.^2;
subjectIs = 1:nSubjects;

load(fullfile(modelPath,options.netString,'RDMs_correlation'),'cnnRDM');
for layerI = 1:numel(cnnRDM)
    cStimRDM_ltv = cnnRDM(layerI).RDM;
    cStimRDM = squareform(cStimRDM_ltv);
    cRDM = repmat(cStimRDM, [4 4]);
    modelRDMs_vec(:,layerI) = squareform(cRDM);
end % layerI


%% fit model to data
for regionI = 1:nRegions
    for foldI = 1:nSubjects
        % perform nonnegative least squares
        subjectI_test = subjectIs(foldI);
        subjectIs_train = setxor(subjectIs,subjectI_test);
        weights = lsqnonneg(double(modelRDMs_vec), mean(dataRDMs_vec(:,subjectIs_train,regionI),2));
        prediction = modelRDMs_vec*weights;
        residuals = dataRDMs_vec(:,subjectI_test,regionI) - prediction;
        predictedRDMs_vec(:,foldI,regionI) = prediction;
    end % subjectI    
end % regionI
modelRDMs_weighted = squeeze(mean(predictedRDMs_vec,2));


%% save models
save(fullfile(savePath,['weightedModelRDMs_',options.netString]),'modelRDMs_weighted');




end

