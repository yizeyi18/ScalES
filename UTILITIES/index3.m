function ind3 = index3(ind1, Ns)
if(ind1 > prod(Ns))
  error('input index is too large');
end

ind3 = zeros(3,1);
ind3(3) = floor((ind1-1)/(Ns(1)*Ns(2)))+1;
tmp = mod(ind1-1, Ns(1) * Ns(2))+1;
ind3(2) = floor((tmp-1) / Ns(1)) + 1;
ind3(1) = mod(tmp-1, Ns(1)) + 1;
