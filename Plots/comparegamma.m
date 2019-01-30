
fileID = fopen('FullLorentzGammaErrorXOR.txt','r');
A = fscanf(fileID,'%s');
B = strsplit(A, '#');

num = length(B)-2;

full_gamma = zeros(num,1);
full_error = zeros(num,1);

for i = 1:num
    n = strsplit(B{i+1}, ';');
    
    full_gamma(i) = str2double(string(n{1}));
    full_error(i) = str2double(string(n{2}));
end

%% 


fileID = fopen('FullDerivLorentzGammaErrorXOR.txt','r');
A = fscanf(fileID,'%s');
B = strsplit(A, '#');

num = length(B)-2;

deriv_gamma = zeros(num,1);
deriv_error = zeros(num,1);

for i = 1:num
    n = strsplit(B{i+1}, ';');
    
    deriv_gamma(i) = str2double(string(n{1}));
    deriv_error(i) = str2double(string(n{2}));
end

%% 


fileID = fopen('SqrtLorentzGammaErrorXOR.txt','r');
A = fscanf(fileID,'%s');
B = strsplit(A, '#');

num = length(B)-2;

sqrt_gamma = zeros(num,1);
sqrt_error = zeros(num,1);

for i = 1:num
    n = strsplit(B{i+1}, ';');
    
    sqrt_gamma(i) = str2double(string(n{1}));
    sqrt_error(i) = str2double(string(n{2}));
end

%% 

a1 = plot(sqrt_gamma, sqrt_error); m1 = "Sqrt Lorentzian Network";
hold on
a2 = plot(full_gamma, full_error); m2 = "Full Lorentzian Network";
a3 = plot(deriv_gamma, deriv_error); m3 = "Full Lorentzian Derivative Network";
legend([a1,a2,a3],[m1,m2,m3])
title("Final error after 10000 iterations for varying values of gamma, for 3 different network types")
xlabel("Gamma")
ylabel("Final Error")
