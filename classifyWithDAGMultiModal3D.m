function scores = classifyWithDAGMultiModal3D(net, data, layertype)
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

if gpuDeviceCount > 0
    net.move('gpu');
end

scores = [];
nused = 0;
nsamples = size(data{1},5);
step = 128;
inix = 1;
endix = min(step, nsamples);

while nused < nsamples
    datachunk1 = data{1}(:,:,:,:,inix:endix);
    
    datachunk2 = data{2}(:,:,:,:,inix:endix);
    if length(data) == 3
        datachunk3 = data{3}(:,:,:,:,inix:endix);
    end
    if strcmp(net.device, 'gpu')
        datachunk1 = gpuArray(datachunk1);
        datachunk2 = gpuArray(datachunk2);
        if length(data) == 3
            datachunk3 = gpuArray(datachunk3);
        end
    end
    
    if ~strcmp(layertype, 'softmax') && ~strcmp(layertype, 'probs')
        net.vars(net.getVarIndex(layertype)).precious = true;
        l = layertype;
    else
        l = 'probs';
    end
    
    if size(data{1}, 3) == 50
        if length(data) == 2
            net.eval({'input', datachunk1(:,:,1:2:50,:,:), 'input2', datachunk1(:,:,2:2:50,:,:), 'input3', datachunk2});
        elseif length(data) == 3
            net.eval({'input', datachunk1(:,:,1:2:50,:,:), 'input2', datachunk1(:,:,2:2:50,:,:), 'input3', datachunk2, 'input4', datachunk3});
        end
    else
        net.eval({'input', datachunk1, 'input3', datachunk2});
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
