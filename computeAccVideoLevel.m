function [acc, estimVidLabs, realVidLabs] = computeAccVideoLevel(estimLabs, realLabs, videoIds)
% [acc, estimVidLabs, realVidLabs] = mj_computeAccVideoLevel(estimLabs, realLabs, videoIds)
% Computes accuracy at video level (several samples per video)
%
% Input:
%  - estimLabs: vector or matrix [nlabels, nsamples]
%  - realLabs: vector
%  - videoIds: vector
%
% Output:
%  - acc: accuracy
%

isRankX = all(size(estimLabs) ~= 1);

uvids = unique(videoIds);
nvids = length(uvids);

tlab = zeros(1,nvids);
rlab = zeros(1,nvids);

for i = 1:nvids 
   idx = videoIds == uvids(i); 
   
   rlab(i) = mode(realLabs(idx)); 
   
   if isRankX % Top-X
      estimLabsR = estimLabs(:,idx);
      tlab(i) = mode(estimLabsR(:)); 
   else % Top-1
      tlab(i) = mode(estimLabs(idx)); 
   end
end

acc = sum(tlab == rlab)/nvids;

% Extra output
estimVidLabs = tlab;
realVidLabs = rlab;
