function y = lorentzianDx0(x, x0, gamma)
y = 4*(x - x0)*(pi / gamma)*(lorentzian(x, x0, gamma).^2);