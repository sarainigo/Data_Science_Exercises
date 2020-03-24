
myOpt=optPrice;
myOpt.assetParam.initPrice=100;
myOpt.assetParam.interest=0;
myOpt.assetParam.volatility=0.4;
myOpt.timeDim.timeVector=1/52:1/52:24/52;
myOpt.priceParam.absTol=0.1;

% PUT OPTION
myOpt.payoffParam.optType={'look'};
myOpt.payoffParam.putCallType={'put'};
[LookBack_Price_Put,lookback_put_out]=genOptPrice(myOpt);
fprintf('Lookback put option price is %1.4f \n',LookBack_Price_Put)

% CALL OPTION
myOpt.payoffParam.putCallType={'call'};
[LookBack_Price_Call,lookback_call_out]=genOptPrice(myOpt);
fprintf('Lookback call option price is %1.4f \n',LookBack_Price_Call)