function [x] = prox_l1(b,lambda)
%% min_x lambda*||x||_1+0.5*||x-b||_F^2
%% input: lambda b
%% output: x
x = max(0,b-lambda)+min(0,b+lambda);