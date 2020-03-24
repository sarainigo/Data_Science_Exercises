
% EXERCISE:

% Stock with initial prize of $40, interest rate of 1%,
% volatility of 40%, monitoring the stock price each week 
% for thirteen weeks.

myOpt=optPrice;
myOpt.assetParam.initPrice=40;
myOpt.assetParam.volatility=0.4;
myOpt.assetParam.interest=0.01;
myOpt.timeDim.timeVector=1/52:1/52:1/4;

% Computation of the price of a up-and-in call option with a strike 
% price of $40 and a barrier of $50 with an absolute error of $ 0.01

% A) IID SAMPLING
myOpt.payoffParam.optType={'upin'};
myOpt.payoffParam.putCallType={'call'};
myOpt.payoffParam.strike=40;
myOpt.payoffParam.barrier=50;
myOpt.priceParam.absTol=0.01;
myOpt.priceParam.relTol=0;
[Up_in,Up_out]=genOptPrice(myOpt);

fprintf('Up and in call option price is %1.4f \n',Up_in)
fprintf('Performance: it took %1.4f seconds, and %5.2f paths \n',Up_out.time,Up_out.nPaths)

% B) IID SAMPLING WITH CONTROL VARIATE

% European call option
Euro=optPrice;
Euro.assetParam.initPrice=40;
Euro.payoffParam.strike=40;
Euro.assetParam.volatility=0.40;
Euro.assetParam.interest=0.01;
Euro.priceParam.relTol=0;
Euro.priceParam.absTol=0.01;
Euro.timeDim.timeVector=1/52:1/52:1/4;
Euro.payoffParam.putCallType={'call'};
Euro.payoffParam.optType={'euro'};

fprintf('European call option exact price is %1.4f \n',Euro.exactPrice)

% Barrier up and in and strike price of 40
Euro_Upin=optPayoff(myOpt);
Euro_Upin.payoffParam = struct('optType',{{'upin','euro'}}, 'putCallType', {{'call','call'}},'strike',([40,40]));
[Up_inEuroPrice, Up_outEuro] = meanMC_g(@(n)YoptPrice_CV(Euro_Upin,n), myOpt.priceParam.absTol,myOpt.priceParam.relTol);

fprintf('Up and in Barrier call option price is %1.4f \n',Up_inEuroPrice)
fprintf('Performance: it took %1.4f seconds, and %5.2f paths \n',Up_outEuro.time,Up_outEuro.n)
comp_t = Up_outEuro.time/Up_out.time
comp_n = Up_outEuro.n/Up_out.nPaths
fprintf('Time Comparison: control variates took %2.4f of the time of simple IID sampling \n',comp_t)
fprintf('N Paths Comparison: control variates took %2.4f of the paths of simple IID sampling \n',comp_n)

% C) INTEGRATION LATICE SAMPLING

Lattice_Upin=optPrice(myOpt);
Lattice_Upin.payoffParam.optType={'upin'};
Lattice_Upin.payoffParam.barrier=50;
Lattice_Upin.priceParam.cubMethod='lattice';
[Up_inLatticePrice, Up_outLattice] = genOptPrice(Lattice_Upin);

fprintf('Up and in Barrier call option price with lattice sampling is %1.4f \n',Up_inLatticePrice)
fprintf('Performance: it took %1.4f seconds, and %5.2f paths \n',Up_outLattice.time,Up_outLattice.nPaths)
comp_t = Up_outLattice.time/Up_out.time
comp_n = Up_outLattice.nPaths/Up_out.nPaths
fprintf('Time Comparison: lattice sampling took %2.4f of the time of simple IID sampling \n',comp_t)
fprintf('N Paths Comparison: lattice sampling took %2.4f of the paths of simple IID sampling\n',comp_n)

% Given the results, we can conclude that the method that takes the least time and paths is the Lattice sampling. The second method that requires less time and paths is the control variate. The method that requires more time and paths is IID sampling.

