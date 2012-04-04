nreps         = [1 4 4];
nelems        = [1 4 4];
bufferratio   = [0.0 0.5 0.5];
asize         = 7.6534;
nat           = 4;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.0   
 0.5      0.0      0.5   
 0.0      0.5      0.5   
];
coefs = coefs + 0.25;

ns_glb  = [24 24 24];
ns_elem = [36 36 36];

% Shift to avoid cut atoms on the element boundary

gen_dgdftin(nreps, nelems, bufferratio, 'Al', ...
  asize, nat, coefs, ns_glb, ns_elem);
