clear; clc;
addpath([pwd, '/my_tools']);
addpath([pwd, '/dataset']);

%% ================Load Dataset===================%%
dataname='MSRC';
load(strcat('dataset/',dataname,'.mat'));
gt = Y;                          % the grount truth for testing
cls_num = length(unique(gt));    % the number of clusters
nV = length(X);                  % the number of views

%% ===============Data preprocessing==============%%
for v = 1:nV
    a = max(X{v}(:));
    X{v} = double(X{v}./a);
end

%% ==============Parameter Initialization=========%%
alpha = 0.025;                    % the hyper-parameter of L1-norm constraint on E(v)
gama = 0.0001;                    % the hyper-parameter of L12-norm constraint on C(v)
anchor_rate = 0.5;                % the anchor rate to all samples
n_sample = size(X{1},1);          % the number of data points
M = fix( n_sample * anchor_rate); % the number of anchors
p = 0.9;                          % the p-value of tensor Schatten p-norm
beta = 10;                        % the initialize parameter of Laplacian rank constraint
weight_vector = ones(1,nV)';      % the defult weight_vector of tensor Schatten p-norm
fid=fopen('MSRC_Result.txt','a'); % build the txt file to save clustering results

%% ==================Training TBGL================%%
[C,kesai] = Train_TBGL(X, cls_num, M, weight_vector, p, alpha, beta, gama);

%% ================Obtain Common Graph=============%%
Common_C = (1/kesai(1))*C{1};
for v = 2: nV
    Common_C = Common_C + (1/kesai(v))*C{v};
end
Sum_kesai = sum(1./kesai);
Common_C = Common_C./Sum_kesai;

%% ==Obtain clustering results based on  the connectivity of Common_C==%%
[Flabel] = my_graphconncomp(Common_C, cls_num, 50);
rs(:,1) = Flabel;

%% =============Measure the performance=============%%
Clu_result = ClusteringMeasure1(Y, rs(:,1))

%% ======================Record=====================%%
fprintf(fid,'lambda: %f ',alpha);
fprintf(fid,'gama12: %f ',gama);
fprintf(fid,'lp: %f ',p);
fprintf(fid,'%g %g %g %g %g %g %g \n ',Clu_result');
fclose(fid);