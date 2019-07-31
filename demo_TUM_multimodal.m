%% Paths.
matmodelcnn = 'models/TUM-GAID-3D-CNN-FUSION.mat';
matimdbof = 'data/tum_of.mat';
matimdbgray = 'data/tum_gray.mat';
matimdbdepth = 'data/tum_depth.mat';
gpuDevice(2);

%% Load net.
load(matmodelcnn);
net.mode = 'test';

%% Load OF test data.
load(matimdbof);
dfactor = 1.0/imdbtest.images.compressFactor;
imdbtest.images.data = single(imdbtest.images.data) * dfactor;
meanval = meta.meanval * dfactor;

%% Load Gray test data.
imdbGray = load(matimdbgray);
imdbGray = imdbGray.imdbtest;
imdbGray.images.data = single(imdbGray.images.data);

%% Load Depth test data.
imdbDepth = load(matimdbdepth);
imdbDepth = imdbDepth.imdbtest;
imdbDepth.images.data = single(imdbDepth.images.data);

%% Balance samples to remove missed samples among modalities.
[imdbtest, imdbGray, imdbDepth] = balanceImdbs(imdbtest, imdbGray, imdbDepth);

%% Change data shape.
% OF
im = single(imdbtest.images.data) - meanval;
im_ = zeros(size(im, 1), size(im, 2), size(im, 3), 1, size(im, 4), class(im));
for i=1:size(im, 4)
    im_(:, :, :, 1, i) = im(:,:,:,i);
end
im = im_;

% Gray
im2 = imdbGray.images.data - meta.meanvalGray;
im_ = zeros(size(im2, 1), size(im2, 2), size(im2, 3), 1, size(im2, 4), class(im2));
for i=1:size(im2, 4)
    im_(:, :, :, 1, i) = im2(:,:,:,i);
end
im2 = im_;

% Depth
im3 = imdbDepth.images.data - meta.meanvalDepth;
im_ = zeros(size(im3, 1), size(im3, 2), size(im3, 3), 1, size(im3, 4), class(im3));
for i=1:size(im3, 4)
    im_(:, :, :, 1, i) = im3(:,:,:,i);
end
im3 = im_;

%% Eval test data.
scoresSMtest = classifyWithDAGMultiModal3D(net, {im, im2, im3}, 'softmax');
[bestScore, best] = max(scoresSMtest);

%% Accuracy at sequence level.
[~, labelsTest] = ismember(imdbtest.images.labels, meta.eqlabs);
[acc_t_sm_seq, estimVidLabs_sm, realVidLabs_sm] = computeAccVideoLevel(best, labelsTest, imdbtest.images.videoId);
fprintf('Accuracy: %.2f \n', 100*acc_t_sm_seq);


