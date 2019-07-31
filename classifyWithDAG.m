function scores = classifyWithDAG(net, data, layertype, imdb, step)
% scores = mj_classifyWithDAG(net, data, layertype, level, imdb)
% Applies a trained net to data.
% Useful to get feature representations of input data.
%
% Input:
%  - net: trained CNN
%  - data: matrix [nrows, ncols, nchannels, nsamples]
%  - layertype: string with type of layer.
%
% Output:
%  - scores: matrix with output data

if ~exist('step', 'var')
    step = 128;
end

if gpuDeviceCount > 0
    net.move('gpu');
end

scores = [];
nused = 0;
if isempty(data)
    nsamples = imdb.images.sizes(4);
else
    nsamples = size(data,4);
end
inix = 1;
endix = min(step, nsamples);

while nused < nsamples
    if isempty(data)
        [im, ~, ~] = loadBatchH5(imdb, inix:endix);
        datachunk = im;
    else
        datachunk = data(:,:,:,inix:endix);
    end
    if strcmp(net.device, 'gpu')
        datachunk = gpuArray(datachunk);
    end
    
    if ~strcmp(layertype, 'softmax')
        net.vars(net.getVarIndex(layertype)).precious = true;
        l = layertype;
    else
        l = 'softmax';
    end
    net.eval({'input', datachunk});
    
    scores_ = net.vars(net.getVarIndex(l)).value ;
    scores_ = squeeze(gather(scores_)) ;
    
    if size(scores_,2) == 1
        scores_ = scores_';
    end
    scores = [scores, scores_];
    
    nused = nused + (endix-inix)+1;
    
    % Update positions
    inix = inix+step;
    endix = min(inix+step-1, nsamples);
end % while
