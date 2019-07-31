%% Paths.
matmodelcnn = 'models/CASIA-RESNET-CNN-GRAY.mat';
matimdbgray = 'data/casia_gray.mat';
gpuDevice(1);

%% Load net.
load(matmodelcnn);
net.mode = 'test';

%% Load Gray test data.
imdbGray = load(matimdbgray);
imdbGray = imdbGray.imdbtest;
imdbGray.images.data = single(imdbGray.images.data) / 255.0;
im = imdbGray.images.data - meta.meanvalGray;

%% Eval test data.
scoresSMtest = classifyWithDAG(net, im, 'probs');
[bestScore, best] = max(scoresSMtest);

%% Accuracy at sequence level.
[~, labelsTest] = ismember(imdbtest.images.labels, meta.eqlabs);
[acc_t_sm_seq, estimVidLabs_sm, realVidLabs_sm] = computeAccVideoLevel(best, labelsTest, imdbtest.images.videoId);
fprintf('Accuracy: %.2f \n', 100*acc_t_sm_seq);


