x = -3:0.001:3;

y0 = lorentzian(x, 0.31, -0.27);
y1 = lorentzian(y0, -0.12, -0.04);
y2 = lorentzian(y1, -2.67, 0.59);

a1 = plot(x, y0); m1 = "First Lorentzian";
hold on
a2 = plot(x, y1); m2 = "Second Lorentzian";
a3 = plot(x, y2); m3 = "Third Lorentzian";
legend([a1, a2, a3], [m1, m2, m3])
hold off