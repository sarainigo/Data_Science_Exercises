
% Since we should generate 1000 instances of x, and c=1/2, half of the X values 
% generated will be accepted on average, so at least there might be 2000 uniform instances. 
% Just in case, we will take 3000.
n = 3000

n_x = 1000
u = unifrnd(0,1,n,1);
z = unifrnd(0,1,n,1);

k = 1
i = 1
x = zeros
while i <= n_x
    % condition of acceptance
    if u(k)<=(6*z(k)*(1-z(k)))/2
        x(i)=z(k);
        i = i+1
    end
    k = k+1
end


