function [imdbOF, imdbGray, imdbDepth] = balanceImdbs(imdbOF_, imdbGray_, imdbDepth_) 
    %% Initialize imdbs.
    imdbOF = imdbOF_;
    imdbGray = imdbGray_;
    imdbDepth = imdbDepth_;
    
    %% Remove different videoIds.
    of = unique(imdbOF_.images.videoId);
    if ~isempty(imdbGray_)
        gray = unique(imdbGray_.images.videoId);
    else
        gray = [];
    end
    
    if ~isempty(imdbDepth_)
        depth = unique(imdbDepth_.images.videoId);
    else
        depth = [];
    end
    
    if isempty(gray)
        diffVideoIds = setdiff(union(of, depth), intersect(of, depth));
    elseif isempty(depth)
	diffVideoIds = setdiff(union(of, gray), intersect(of, gray));
    else
	diffVideoIds = setdiff(union(of, union(gray, depth)), intersect(of, intersect(gray, depth)));
    end

    for i=1:length(diffVideoIds)
        if isempty(find(of == diffVideoIds(i))) && length(of) < max(of) 
            pos = find(imdbOF_.images.videoId == diffVideoIds(i)+1);
            imdbOF.images.videoId(pos(1):end) = imdbOF.images.videoId(pos(1):end) - 1;
        else
            if ~isempty(find(gray == diffVideoIds(i)))
                pos = find(imdbGray.images.videoId == diffVideoIds(i));
                imdbGray.images.data(:,:,:,pos) = [];
                imdbGray.images.labels(pos) = [];
                imdbGray.images.videoId(pos) = [];
                imdbGray.images.set(pos) = [];
                imdbGray.images.gait(pos) = [];
                if isfield(imdbGray.images, 'mirrors')
                    imdbGray.images.mirrors(pos) = [];
                end
            end
            
            if ~isempty(find(depth == diffVideoIds(i)))
                pos = find(imdbDepth.images.videoId == diffVideoIds(i));
                imdbDepth.images.data(:,:,:,pos) = [];
                imdbDepth.images.labels(pos) = [];
                imdbDepth.images.videoId(pos) = [];
                imdbDepth.images.set(pos) = [];
                imdbDepth.images.gait(pos) = [];
                if isfield(imdbDepth.images, 'mirrors')
                    imdbDepth.images.mirrors(pos) = [];
                end
            end
        end
    end
       
    if ~isempty(imdbGray_)
        imdbGray.images.labels = imdbOF.images.labels;
        imdbGray.images.videoId = imdbOF.images.videoId;
        imdbGray.images.set = imdbOF.images.set;
        if isfield(imdbOF.images, 'mirrors') && isfield(imdbGray.images, 'mirrors')
            imdbGray.images.mirrors = imdbOF.images.mirrors;
        end
        imdbGray.images.gait = imdbOF.images.gait;
        imdbGray.images.data = zeros(size(imdbGray.images.data, 1), size(imdbGray.images.data, 2), size(imdbGray.images.data, 3), length(imdbOF.images.labels), class(imdbGray.images.data));
    end
    
    if ~isempty(imdbDepth_)
        imdbDepth.images.labels = imdbOF.images.labels;
        imdbDepth.images.videoId = imdbOF.images.videoId;
        imdbDepth.images.set = imdbOF.images.set;
        if isfield(imdbOF.images, 'mirrors') && isfield(imdbDepth.images, 'mirrors')
            imdbDepth.images.mirrors = imdbOF.images.mirrors;
        end
        imdbDepth.images.gait = imdbOF.images.gait;
        imdbDepth.images.data = zeros(size(imdbDepth.images.data, 1), size(imdbDepth.images.data, 2), size(imdbDepth.images.data, 3), length(imdbOF.images.labels), class(imdbDepth.images.data));
    end
    
    posGray = 1;
    posDepth = 1;
    for posOF=1:length(imdbOF.images.labels)
        %% Gray.
        condition = true;
        if ~isempty(imdbGray_) && posGray < length(imdbGray_.images.labels)
            condition = condition && (imdbOF.images.labels(posOF) == imdbGray_.images.labels(posGray));
            condition = condition && (imdbOF.images.videoId(posOF) == imdbGray_.images.videoId(posGray));
            condition = condition && (imdbOF.images.gait(posOF) == imdbGray_.images.gait(posGray));

            if condition
                % Same sample
                imdbGray.images.data(:, :, :, posOF) = imdbGray_.images.data(:, :, :, posGray);
            else
                % Different sample
                keepRunning = true;
                while keepRunning && posGray < length(imdbGray_.images.labels)
                    condition = true;
                    posGray = posGray + 1;
                    condition = condition && (imdbOF.images.labels(posOF) == imdbGray_.images.labels(posGray));
                    condition = condition && (imdbOF.images.videoId(posOF) == imdbGray_.images.videoId(posGray));
                    condition = condition && (imdbOF.images.gait(posOF) == imdbGray_.images.gait(posGray));
                    
                    if condition
                        keepRunning = false;
                        posGray = posGray - 1;
                    end
                end
            end

            posGray = posGray + 1;           
        end

        %% Depth
        condition = true;
        if ~isempty(imdbDepth_) && posDepth < length(imdbDepth_.images.labels)
            condition = condition && (imdbOF.images.labels(posOF) == imdbDepth_.images.labels(posDepth));
            condition = condition && (imdbOF.images.videoId(posOF) == imdbDepth_.images.videoId(posDepth));
            condition = condition && (imdbOF.images.gait(posOF) == imdbDepth_.images.gait(posDepth));

            if condition
                % Same sample
                imdbDepth.images.data(:, :, :, posOF) = imdbDepth_.images.data(:, :, :, posDepth);
            else
                % Different sample
                keepRunning = true;
                while keepRunning && posDepth < length(imdbDepth_.images.labels)
                    condition = true;
                    posDepth = posDepth + 1;
                    condition = condition && (imdbOF.images.labels(posOF) == imdbDepth_.images.labels(posDepth));
                    condition = condition && (imdbOF.images.videoId(posOF) == imdbDepth_.images.videoId(posDepth));
                    condition = condition && (imdbOF.images.gait(posOF) == imdbDepth_.images.gait(posDepth));

                    if condition
                        keepRunning = false;
                        posDepth = posDepth - 1;
                    end
                end
            end

            posDepth = posDepth + 1;
        end
    end
end
