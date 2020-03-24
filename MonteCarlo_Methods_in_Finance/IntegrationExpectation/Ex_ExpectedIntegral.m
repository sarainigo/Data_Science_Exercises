
% Function definition
Yrand=@(n)f(randn(n,2));
muhat=meanMC_g(Yrand,0.02,0);