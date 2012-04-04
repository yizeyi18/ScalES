nreps         = [1 2 2];
nelems        = [1 4 4];
bufferratio   = [0.0 0.5 0.5];
asize         = 10.2612;
nat           = 8;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.0   
 0.5      0.0      0.5  
 0.0      0.5      0.5 
 0.25     0.25     0.25
 0.75     0.75     0.25
 0.75     0.25     0.75
 0.25     0.75     0.75 
];
coefs = coefs + 0.125;
%coefs = coefs + 0.001;

ns_glb  = [32 32 32];
ns_elem = [40 24 24];


% Shift to avoid cut atoms on the element boundary

gen_dgdftin(nreps, nelems, bufferratio, 'Si', ...
  asize, nat, coefs, ns_glb, ns_elem, 0.0,'dg');
