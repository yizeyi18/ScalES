nreps         = [1 1 1];
nelems        = [1 1 1];
bufferratio   = [0.0 1.0 1.0];
asize         = [10.00 8.0540  4.65];
nat           = 4;
coefs = [
 0      0        0
 0      1/6      1/2
 0      1/2      1/2
 0      2/3      0
];
shift = [0.5 1/6 0.25];
coefs = coefs + repmat(shift, 4, 1);

% ns_glb  = [48 40 24];
% ns_elem = [64 56 36];

ns_glb  = [32 32 16];
ns_elem = [48 48 24];

% Shift to avoid cut atoms on the element boundary

gen_dgdftin(nreps, nelems, bufferratio, 'C', ...
  asize, nat, coefs, ns_glb, ns_elem, 0.0);
