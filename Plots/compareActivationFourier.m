x = -3:0.001:3;

y_lorentzian = lorentzian(x, 0, 1);
y_lorentziandx = lorentzianDx(x, 0, -1);
y_sigmoid = 1./(1 + exp(-x));
y_tanh = tanh(x);
y_relu = max(0, x);

f_lorentzian = abs(fft(y_lorentzian));
f_lorentziandx = abs(fft(y_lorentziandx));
f_sigmoid = abs(fft(y_sigmoid));
f_tanh = abs(fft(y_tanh));
f_relu = abs(fft(y_relu));

poly_degree = 5;

p_lorentzian = fliplr(polyfit(x, y_lorentzian, poly_degree));
p_lorentziandx = fliplr(polyfit(x, y_lorentziandx, poly_degree));
p_sigmoid = fliplr(polyfit(x, y_sigmoid, poly_degree));
p_tanh = fliplr(polyfit(x, y_tanh, poly_degree));
p_relu = fliplr(polyfit(x, y_relu, poly_degree));

subplot(3,1,1)

a1 = plot(x, y_lorentzian); m1 = "Lorentzian";
hold on
a2 = plot(x, y_lorentziandx); m2 = "Lorentzian Derivative";
a3 = plot(x, y_sigmoid); m3 = "Sigmoid";
a4 = plot(x, y_tanh); m4 = "Tanh";
a5 = plot(x, y_relu); m5 = "reLU";

title("Plot of Activation Functions")

legend([a1,a2,a3,a4,a5],[m1,m2,m3,m4,m5])

hold off

subplot(3,1,2)

a1 = plot(f_lorentzian(1:10));
hold on
a2 = plot(f_lorentziandx(1:10));
a3 = plot(f_sigmoid(1:10));
a4 = plot(f_tanh(1:10));
a5 = plot(f_relu(1:10));

xlabel("Frequency")
ylabel("|X(f)|")
title("Fourier Transform of Activation Functions")

%legend([a1,a2,a3,a4,a5],[m1,m2,m3,m4,m5])

hold off

subplot(3,1,3)

a1 = plot(1:5, p_lorentzian(2:length(p_lorentzian)));
hold on
a2 = plot(1:5, p_lorentziandx(2:length(p_lorentzian)));
a3 = plot(1:5, p_sigmoid(2:length(p_lorentzian)));
a4 = plot(1:5, p_tanh(2:length(p_lorentzian)));
a5 = plot(1:5, p_relu(2:length(p_lorentzian)));
xticks([1 2 3 4 5])
xlabel("Polynomial powers")
ylabel("Coefficients")
title("Coefficients of Polynomial of degree 5 fit to Activation Functions")

%legend([a1,a2,a3,a4,a5],[m1,m2,m3,m4,m5])

hold off




