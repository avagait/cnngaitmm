%% Paths.
matmodelcnn = 'models/TUM-GAID-3D-CNN-OF.mat';
matimdbof = 'data/tum_of.mat';
gpuDevice(2);

%% Load net.
load(matmodelcnn);
net.mode = 'test';

%% Load OF test data.
load(matimdbof);
dfactor = 1.0/imdbtest.images.compressFactor;
imdbtest.images.data = single(imdbtest.images.data) * dfactor;
meanval = meta.meanval * dfactor;

%% Change data shape.
% OF
im = single(imdbtest.images.data) - meanval;
im_ = zeros(size(im, 1), size(im, 2), size(im, 3), 1, size(im, 4), class(im));
for i=1:size(im, 4)
    im_(:, :, :, 1, i) = im(:,:,:,i);
end
im = im_;

%% Eval test data.
scoresSMtest = classifyWithDAG3D(net, im, 'softmax');
[bestScore, best] = max(scoresSMtest);

%% Accuracy at sequence level.
[~, labelsTest] = ismember(imdbtest.images.labels, meta.eqlabs);
[acc_t_sm_seq, estimVidLabs_sm, realVidLabs_sm] = computeAccVideoLevel(best, labelsTest, imdbtest.images.videoId);
fprintf('Accuracy: %.2f \n', 100*acc_t_sm_seq);


