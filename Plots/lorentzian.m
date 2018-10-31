function y = lorentzian(x, x0, lambda)
y = (0.5*lambda) ./ (pi * ((x - x0).^2 + (0.5*lambda).^2));