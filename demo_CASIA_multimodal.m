%% Paths.
matmodelcnn = 'models/CASIA-3D-CNN-FUSION.mat';
matimdbof = 'data/casia_of.mat';
matimdbgray = 'data/casia_gray.mat';
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
imdbGray.images.data = single(imdbGray.images.data) / 255.0;

%% Balance samples to remove missed samples among modalities.
[imdbtest, imdbGray, ~] = balanceImdbs(imdbtest, imdbGray, []);

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

%% Eval test data.
scoresSMtest = classifyWithDAGMultiModal3D(net, {im, im2}, 'softmax');
[bestScore, best] = max(scoresSMtest);

%% Accuracy at sequence level.
[~, labelsTest] = ismember(imdbtest.images.labels, meta.eqlabs);
[acc_t_sm_seq, estimVidLabs_sm, realVidLabs_sm] = computeAccVideoLevel(best, labelsTest, imdbtest.images.videoId);
fprintf('Accuracy: %.2f \n', 100*acc_t_sm_seq);


