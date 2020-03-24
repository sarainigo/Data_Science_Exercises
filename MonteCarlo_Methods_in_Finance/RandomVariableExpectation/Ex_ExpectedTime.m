% EXERCISE 1: Expected time required to wait for a bus if you arrive at the bus stop at a random time

% nMc: number of MC simulations
nMC=1e6;
% We create uniform variable T 
T=unifrnd(-1,14,nMC,1);
% We create 2 uniform variables U1 y U2 
U=unifrnd(-1,2,nMC,2);

% I1 records with 1 when we arrive earlier that the first bus, and with 0 when not
I1=T<=U(:,1);
% We record in T1 only the times when we arrive earlier that the first bus
T1=T(I1);
% We record in U1 only the bus arrivals when we arrive earlier that the first bus
U1=U(I1,1);

% I2 records with 1 when we arrive later that the first bus, and with 0 when not
I2=T>U(:,1);
% We record in T2 only the times when we arrive later that the first bus
T2=T(I2);
% We record in U2 only the bus arrivals when we arrive later that the first bus
U2=U(I2,2);

% Calculation of the nMC samples of the waiting time: Two cases
% Case 1: We arrive earlier than the first bus
W1=U1-T1;
% Case 2: We arrive later than the first bus
W2=15+U2-T2;

% Monte Carlo Approximation
sample_mean=(sum(W1)+sum(W2))/nMC;
% Exact value
value_exact=151/20;
% Absolute error of the Monte Carlo Approximation
Error=abs(value_exact-sample_mean);


% EXERCISE 3: Use Central Limit Theorem confidence intervals to compute the answer to 1) by Monte Carlo simulation to an absolute tolerance of 1 second with 99% confidence

% Calculation of the sample variance
W = cat(1,W1,W2);
sample_var = sum((W-sample_mean).^2)/(nMC-1);

% Central Limit Theorem confidence intervals with 99% confidence
low_int = sample_mean - (2.58*sample_var)/sqrt(nMC);
high_int = sample_mean + (2.58*sample_var)/sqrt(nMC);


% EXERCISE 4: For the situation in problem 1, construct a confidence interval for the probability that the time to wait of a bus is greater than 8 minutes using 10^6 Monte Carlo samples and binomialCI.

% If the time to wait a bus is greater than 8 minutes, two conditions must be met
% Condition 1: T>U(:,1)-> we should arrive later that the first bus
% Condition 2: (15+U(:,2)-T)>8 -> we should wait more than 8 minutes for the second bus

% suc records with 1,0 when both conditions are met
suc = (T>U(:,1)) & ((15+U(:,2)-T)>8)
% number of successes in the sample
nsuc = sum(suc)

% Confidence intervals with 99% confidence
CI = binomialCI(nMC,nsuc, 0.01);

