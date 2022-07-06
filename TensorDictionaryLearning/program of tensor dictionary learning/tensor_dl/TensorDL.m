function [ clean_img, basis, idx, tmp, info, CDIC] = TensorDL( noisy_img, params )

%==========================================================================
% TensorDL removes noise from multi-spectral imagery (MSI) via block matching
%   and tensor factorization/decomposition.
%
% Syntax:
%   [ clean_img, basis, idx, cores, info, CIDC ] = TensorDL( noisy_img, params );
%
% Input arguments:
%   noisy_img...the MSI in a 3D array of size height x width x nbands.
%   params......an option structure whose fields are as follows:
%       peak_value: the upper bound of dynamic range. (required)
%       block_sz: a 1-by-2 integer vector that indicates the size of blocks. 
%               If block_sz is a scalar, the height and width of blocks will
%               be the same. (default [8, 8] or estimated by nsigma)
%       overlap_sz: a 1-by-2 integer vector that indicates the size of
%               overlaps. If overlap_sz is a scalar, the vertical and
%               horizontal overlap size will be the same. (default
%               block_sz/2)
%       spect_dim: truncated dimension of spectral bands. It is an integer
%               in [1, size(noisy_img, 3)]. If it's not specified, it will
%               be estimated by MDL
%       nclusters: number of clusters in block matching. (default value is
%               computed from the total number of blocks)
%       predenoise: a logical parameter to tell whether a pre-denoise step
%               should be taken to improve the performance of block
%               matching. (default 1)
%       nsigma: standard deviation of noise, used to estimate the size 
%               and standard deviation of Gaussianin template in
%               predenoising, and to determine the size of blocks in which
%               case block_sz is not specified
%       dic_num: the dictionary number that we want to concatenate, if not 
%                specified, the default number is 1
%
% Output arguments:
%   clean_img...the denoised imagery. The input imagery can be of any class
%               (integer or float), but the output imagery will be a double
%               array of the same size and similar dynamic range.
%   basis.......a 4D array. It denotes the combination of basis of spanned
%               spece in in several groups.
%   idx.........the indices of blocks in clustering.
%   cores.......core tensors of each group.
%   info........a structure which contains regularized information in the
%               input argument params.
%   CDIC........structure of concatenated dictionaries U1 U2 U3, to get 
%               U1,U2,U3, please use CDIC.U1 CDIC.U2 CDIC.U3
%
% by Yi Peng
%==========================================================================

sz = size(noisy_img);
if ndims(noisy_img) ~= 3
    if ndims(noisy_img) == 2
        error('Only one band exist!.');
    else
        error(['Wrong imagery data format!\n' ...
            'Make sure it is a  height x width x nbands array.']);
    end
end

if ~exist('params', 'var')
    params = [];
end

if ~isfield(params, 'block_sz')
    if isfield(params, 'nsigma')
        if ~isfield(params, 'peak_value')
            error(['The peak value of spectral image is required while using' ...
            'noise standard deviation to estimate size of patches!']);
        end
        local_std = LocalStd(noisy_img);
        if (local_std^2 - params.nsigma^2)/params.peak_value^2 < 0.01
            bparams.block_sz = [8, 8];
        else
            %bparams.block_sz = [5, 5];
            bparams.block_sz = [8, 8];
        end
    else
        bparams.block_sz = [8, 8];
    end
elseif isscalar(params.block_sz)
    bparams.block_sz = repmat(params.block_sz, 1, 2);
else
    bparams.block_sz = [params.block_sz(1), params.block_sz(2)];
end

if ~isfield(params, 'overlap_sz')
    bparams.overlap_sz = fix(bparams.block_sz/2);
elseif isscalar(params.overlap_sz)
    bparams.overlap_sz = repmat(params.overlap_sz, 1, 2);
else
    bparams.overlap_sz = [params.overlap_sz(1), params.overlap_sz(2)];
end
bparams.block_num = floor((sz(1:2) - bparams.overlap_sz)./(bparams.block_sz - bparams.overlap_sz));
info = bparams;
noisy_blocks = ExtractBlocks( noisy_img, bparams );

if isfield(params, 'nclusters')
    nclusters = params.nclusters;
else
    nclusters = ceil(prod(bparams.block_num)/100);
end
info.nclusters = nclusters;

if isfield(params, 'spect_dim')
    if params.spect_dim < 1 || params.spect_dim > sz(3) ...
            || params.spect_dim - round(params.spect_dim) ~= 0
        error('The truncated dimension of spectral bands is illegal!');
    else
        useMDL4specdim = false;
        info.spect_dim = params.spect_dim;
    end
else
    useMDL4specdim = true;
    info.spect_dim = 'MDL';
end

% block matching (if needed, pre-denoise via bandwise gaussian filtering first)
if ~isfield(params, 'predenoise') || params.predenoise
    info.predenoise = 1;
    % fprintf('Pre-denoising for better block matching...\n');
    if ~isfield(params, 'nsigma')
        hsize = round(bparams.block_sz(1)/2);
    else
        if ~isfield(params, 'peak_value')
            error(['The peak value of spectral image is required while using' ...
            'noise standard deviation to estimate size of Gaussian filter!']);
        end
        epsilon = (params.peak_value*0.05)^2;
        hsigma = params.nsigma^2/4/pi/epsilon;
        hsize = min(ceil(hsigma*6), round(bparams.block_sz(1)/2));
    end
    hsigma = hsize/6;
    H = fspecial('gaussian', hsize, hsigma);
    predenoised_img = zeros(sz);
    for k = 1:sz(3)
        predenoised_img(:, :, k) = filter2(H, noisy_img(:, :, k));
    end
    predenoised_blocks = ExtractBlocks(predenoised_img, bparams);    
    X = reshape(predenoised_blocks, prod(bparams.block_sz)*sz(3), size(noisy_blocks, 4))';
else
    info.predenoise = 0;
    X = reshape(noisy_blocks, prod(bparams.block_sz)*sz(3), size(noisy_blocks, 4))';    
end
fkmeans_opt.careful = 1;
fprintf('Matching similar blocks...\n');
% idx = fkmeans(X, nclusters, fkmeans_opt);
%  [idx,~] = kmeans(X,nclusters);
idx = fkmeans(X, nclusters, fkmeans_opt);

[count,center] = hist(idx,1:nclusters);
[out,odr] = sort(count,'descend');
center = center(odr);
Unum=1;

% denoise each group by tensor factorization
% fprintf('Denoising each group of blocks...\n');
clean_blocks = zeros(size(noisy_blocks));
natoms = 4;     % max number of atoms that are displayed in each group
basis = zeros([bparams.block_sz, sz(3), nclusters*natoms]);
cores = cell(1, nclusters);

if ~isfield(params, 'dic_num')
    dic_num = 1;
else
    dic_num = params.dic_num;
end
%for k = 1:2%nclusters
for k = center(1:dic_num)%clusters chosen
    R = [bparams.block_sz, 0, 0];
    nblocks = numel(find(idx==k));
    matched_blocks = tensor(noisy_blocks(:, :, :, idx==k));
    if useMDL4specdim
        flat_blocks = double(tenmat(matched_blocks, 3))';
        score = DimDetectMDL(flat_blocks);
        [~, R(3)] = min(score);
    else
        R(3) = params.spect_dim;
    end
    if nblocks > 1
        flat_blocks = double(tenmat(matched_blocks, 4))';
        score = DimDetectAIC(flat_blocks);
        [~, R(4)] = min(score);
    else
        R(4) = [];
    end
    tmp = TuckerALS(matched_blocks, R, 'printitn', 0);
    
    U1{Unum} = tmp.U{1};
    U2{Unum} = tmp.U{2};
    U3{Unum} = tmp.U{3};
    Unum = Unum + 1;
    
    cores{k} = tmp.core;
    clean_blocks(:, :, :, idx==k) = double(tmp);
    if nblocks > 1
        btmp = double(ttm(tmp.core, tmp.U, -4));
        ntmp = min(R(4), natoms);
        basis(:, :, :, natoms*k-3:natoms*k-natoms+ntmp) = btmp(:, :, :, 1:ntmp);
    else
        basis(:, :, :, natoms*k-3) = double(tmp);
    end
end

clean_img = JointBlocks(clean_blocks, bparams);
CDIC.U1 = cat(2,U1{:});
CDIC.U2 = cat(2,U2{:});
CDIC.U3 = U3{1};
end

%% function for calculating average standard deviation
function s = LocalStd( img )

    bparams.block_sz = [32, 32];
    bparams.overlap_sz = [16, 16];
    blocks = ExtractBlocks(img, bparams);

    s = 0;
    for i = 1:size(blocks, 3)
        for j = 1:size(blocks, 4);
            s = s + std2(blocks(:, :, i, j));
        end
    end
    s = s/size(blocks, 3)/size(blocks, 4);

end