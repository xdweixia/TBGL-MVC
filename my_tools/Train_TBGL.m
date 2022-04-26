function [C,kesai] = Train_TBGL(X, cls_num, M, weight_vector, p, alpha, beta, gama)

nV = length(X); N = size(X{1},1);

%% ==============Variable Initialization=========%%
for k = 1:nV
    C{k} = zeros(N, M);
    W{k} = zeros(N, M);
    Y2{k} = zeros(N, M);
    J{k} = zeros(N, M);
    E{k} = zeros(N, M);
    Q{k} = zeros(N, M);
    Z{k} = zeros(N, M);
    QQ{k} = zeros(N, M);
    HS{k}= zeros(N, M);
    Y3{k} = zeros(N, M);
end
kesai = repmat(1/nV, [1,nV]);

%%
disp('--------------Anchor Selection and Bipartite graph Construction----------');
tic;
opt1. style = 1;
opt1. IterMax =50;
opt1. toy = 0;
[~, B] = My_Bipartite_Con(X,cls_num,0.5, opt1,10);
toc;

%% =====================  Initialization =====================
M = size(B{1},2); nV = length(X);   N = size(X{1},1);
F = zeros(N+M,cls_num);    sX = [N, M, nV];
Isconverg = 0; iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 1.1;
rho = 1e-4; max_rho = 10e12; pho_rho = 1.1;
mu3 = 10e-5;

Dc{1} = diag(sum(B{1},1)+eps);
ZZ = (1/kesai(1))*B{1}*Dc{1}^-0.5;
for v = 2: nV
    Dc{v} = diag(sum(B{v},1)+eps);
    ZZ = ZZ + (1/kesai(v))*B{v}*Dc{v}^-0.5;
end
[uu, ~, vv] = svd(ZZ);
F = 1/sqrt(2) * [uu(:, 1:cls_num); vv(:, 1:cls_num)];
%% =====================Optimization=====================
while(Isconverg == 0)
    %% solve C{v}
    for v =1:nV
        Q{v} = (B{v} - E{v} + W{v}/mu);
        P{v} = J{v} + Y2{v}/rho;
        H{v} = Dc{v}^-0.5*F(N+1:end,:)*F(1:N,:)'*(1/kesai(v));
        HH{v} = H{v}';
        PHS{v} = HS{k}- Y3{v}/mu3;
    end
    y = zeros(N, M);
    for v = 1:nV
        for i = 1:N
            y(i, :) = mu*Q{v}(i, :) + rho*P{v}(i, :) + mu3*PHS{v}(i, :) + 2 * beta *HH{v}(i, :);
            MM = (y(i, :))/ (rho + mu + mu3);
            C{v}(i, :) = EProjSimplex_new(MM, 1);
        end
    end

    %%  solve J{v}
    for v =1:nV
        QQ{v}=(C{v} - Y2{v}/rho);
    end
    Q_tensor = cat(3,QQ{:,:});
    Qg = Q_tensor(:);
    [myj, ~] = wshrinkObj_weight_lp(Qg, weight_vector./rho,sX, 0,3,p);
    J_tensor = reshape(myj, sX);
    for k=1:nV
        J{k} = J_tensor(:,:,k);
    end
    %% solve O{v}, i.e., HS{k}
    for v =1:nV
        HS{v} = solve_L12norm(C{v}+Y3{v}./mu3, 2*gama/mu3);
    end

    %% solve E{v}
    for v =1:nV
        P{v}=(B{v} - C{v} + W{v}/mu);
        E{v} = prox_l1(P{v},alpha/mu);
    end

    %% solve P
    clear ZZ;
    clear Dc;
    Dc{1} = diag(sum(C{1},1)+eps);
    ZZ = (1/kesai(1))*C{1}*Dc{1}^-0.5;
    for v = 2: nV
        Dc{v} = diag(sum(C{v},1)+eps);
        ZZ = ZZ + (1/kesai(v))*C{v}*Dc{v}^-0.5;
    end
    [uu, ev1, vv] = svd(ZZ);
    U1 = uu(:,1:cls_num); V1 = vv(:,1:cls_num);
    U = sqrt(2)/2*U1; V = sqrt(2)/2*V1;
    ev = diag(ev1);
    U_old = U; V_old = V;
    fn1 = sum(ev(1:cls_num));
    fn2 = sum(ev(1:cls_num+1));
    if fn1 < cls_num-0.0000001
        beta = 2*beta;
    elseif fn2 > cls_num+1-0.0000001
        beta = beta/2;   U = U_old; V = V_old;
    else
        break
    end
    F=[U; V];

    %% solve kesai
    for v = 1:nV
        LDc{v} = diag(sum(C{v},1)+eps);
        LDr{v} = diag(sum(C{v},2));
        LD{v} = blkdiag(LDr{v},LDc{v});
        tmp1 = zeros(N+M);
        tmp1(1:N,N+1:end) = C{v};
        tmp1(N+1:end,1:N) = C{v}';
        LW{v} = tmp1;
        L{v} = eye(N+M) - (LD{v}^-0.5) * LW{v} * (LD{v}^-0.5);
    end
    for v = 1:nV
        h(v) = sqrt(trace(F'*L{v}*F));
    end
    %
    kesai = h./sum(h);

    %% solve Y1 and  penalty parameters
    for k=1:nV
        W{k} = W{k} + mu*(B{k} - E{k}-C{k});
        Y2{k} = Y2{k} + rho*(J{k}-C{k});
        Y3{k} = Y3{k} + mu3*(C{k}-HS{k});
    end
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
    mu3 = min(mu3*pho_mu, max_mu);

    %% ==============Max Training Epoc==============%%
    if (iter>50)
        Isconverg  = 1;
    end
    iter = iter + 1;
end
