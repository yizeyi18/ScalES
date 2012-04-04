function idx = getindex3(g, Ns)

idx = zeros(3,1);

idx(3) = ceil(g/(Ns(1)*Ns(2)));
g = mod(g,Ns(1)*Ns(2));
idx(2) = ceil(g/Ns(1));
g = mod(g,Ns(1));
idx(1) = g;