function scores = classifyWithDAG3D(net, data, layertype, imdb, step)
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
    nsamples = size(data,5);
end
inix = 1;
endix = min(step, nsamples);

while nused < nsamples
    if isempty(data)
        [im, ~, ~] = loadBatchH5(imdb, inix:endix);
        datachunk = zeros(size(im, 1), size(im, 2), size(im, 3), 1, size(im, 4), class(im));
        for i=1:size(im, 4)
            datachunk(:, :, :, 1, i) = im(:,:,:,i);
        end
    else
        datachunk = data(:,:,:,:,inix:endix);
    end
    if strcmp(net.device, 'gpu')
        datachunk = gpuArray(datachunk);
    end
    
    if ~strcmp(layertype, 'softmax') && ~strcmp(layertype, 'probs')
        net.vars(net.getVarIndex(layertype)).precious = true;
        l = layertype;
    else
        l = 'probs';
    end
    
    if size(datachunk, 3) == 50
        net.eval({'input', datachunk(:,:,1:2:50,:,:), 'input2', datachunk(:,:,2:2:50,:,:)});
    else
        net.eval({'input', datachunk});
    end
    
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

