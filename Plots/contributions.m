fileID = fopen('FullLorentzGammaErrorXOR.txt','r');
A = fscanf(fileID,'%s');
B = strsplit(A, '#');

num = length(B)-2;

gamma = zeros(num,1);
error = zeros(num,1);

input_size = 2;
hidden_layer_size = 4;
output_size = 1;

first_weights = zeros(num, hidden_layer_size, input_size);
second_weights = zeros(num, output_size, hidden_layer_size);

for i = 1:num
    n = strsplit(B{i+1}, ';');
    
    gamma(i) = str2double(string(n{1}));
    error(i) = str2double(string(n{2}));
    weights_str = strsplit(string(n{3}), 'array');
    weights_str = strrep(weights_str, '[', '');
    weights_str = strrep(weights_str, ']', '');
    weights_str = strrep(weights_str, '(', '');
    weights_str = strrep(weights_str, ')', '');
    
    first_weights_str = strsplit(weights_str{2}, ',');
    second_weights_str = strsplit(weights_str{3}, ',');
    
    for j = 1:input_size
        for k = 1:hidden_layer_size
            first_weights(i,k,j) = str2double(first_weights_str{((k*input_size)-1) + (j-1)});
        end
    end
    
    for j = 1:hidden_layer_size
        %for k = 1:output_size
        %second_weights(i,j) = str2double(second_weights_str{((j*output_size)-1) + (k-1)});
        %end
        second_weights(i,j) = str2double(second_weights_str{j});
    end
end

inputs = [0,0;0,1;1,0;1,1];
targets = [0 ; 1 ; 1 ; 0];

input = repmat(inputs(1,:), hidden_layer_size, 1);
layer = sum(lorentzian(input, squeeze(first_weights(1,:,:)), gamma(1)*ones(hidden_layer_size,input_size)), 2);

output = lorentzian(layer', second_weights(1,:), gamma(1)*ones(output_size,hidden_layer_size));

layers = zeros(num, hidden_layer_size, length(targets));
outputs = zeros(num, hidden_layer_size, length(targets));

for i = 1:num
    for j = 1:length(inputs)
        input = repmat(inputs(j,:), hidden_layer_size, 1);
        layer = sum(lorentzian(input, squeeze(first_weights(i,:,:)), gamma(i)*ones(hidden_layer_size,input_size)), 2);

        layers(i,j,:) = layer';
        outputs(i,j,:) = lorentzian(layer', second_weights(i,:), gamma(i)*ones(output_size,hidden_layer_size));
    end
end

g = 55;

x = 0:0.001:4;

for i = 1:hidden_layer_size
    subplot(hidden_layer_size, 2, (2*i)-1)
    y = lorentzian(x, second_weights(g,i), gamma(g));
    plot(x,y)
    hold on
    a1 = scatter(layers(g,1,i), outputs(g,1,i), '*');
    a2 = scatter(layers(g,2,i), outputs(g,2,i), '+');
    a3 = scatter(layers(g,3,i), outputs(g,3,i), 'x');
    a4 = scatter(layers(g,4,i), outputs(g,4,i));
    title(sprintf('Node %d', i));
end

legend([a1,a2,a3,a4], {'0,0','0,1','1,0','1,1'})

subplot(hidden_layer_size, 2, [2,4,6,8])
c = categorical({'0,0','0,1','1,0','1,1'});
h = bar(c,squeeze(outputs(g,:,:)), 'stacked');
legend(h, {'Node 1','Node 2','Node 3','Node 4'})







