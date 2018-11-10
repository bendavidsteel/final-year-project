function y = lorentzian(x, x0, gamma)
y = (0.5*gamma) ./ (pi * ((x - x0).^2 + (0.5*gamma).^2));