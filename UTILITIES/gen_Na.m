nreps         = [1 1 4];
nelems        = [1 1 4];
bufferratio   = [1.0 1.0 1.0];
asize         = 7.9994;
nat           = 2;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.5   
];
coefs = coefs + 0.25;

ns_glb  = [12 12 12];
ns_elem = [20 20 20];

% Shift to avoid cut atoms on the element boundary

gen_dgdftin(nreps, nelems, bufferratio, 'Na', ...
  asize, nat, coefs, ns_glb, ns_elem, 0.0, 'global');