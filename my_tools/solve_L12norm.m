function [E_new]= solve_L12norm(E, lembda)
%%标准型：lembda||E_new||_{1,2}^2+||E_new-E||_F^2
%%输入：lembda、E
%%输出：E_new
%%%%%%% 按列求解 E_j  %%%%%%   
[m,n] = size(E);
E_new = zeros(m,n);
%%%%%%%   用于求解L12范数 %%%%%%%
for k=1:n
    e = E(:,k);
    [tao, mu] = search_tao_mu(e, lembda);
    e_temp = abs(e) - lembda*tao/(1.0 + lembda*tao)*mu;
    e_temp(find(e_temp<0)) = 0;
    E_new(:,k) = sign(e).*e_temp;
end

function [tao,mu] = search_tao_mu(e, lembda)
%
d = length(e);
e_abs = abs(e);
[temp,S] = sort(e_abs,'descend');
% Initialize
tao = d;
mu = 1.0/d*sum(e_abs);
while (tao>1) && (e_abs(S(tao)) - lembda*tao/(1+lembda*tao)*mu <0)
    mu = tao/(tao-1)*mu - 1.0/(tao-1)*e_abs(S(tao));
    tao = tao-1;
end



