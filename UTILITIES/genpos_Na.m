nreps         = [1 4 4];
asize         = 7.9994; 
nat           = 2;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.5   
];
coefs = coefs + 0.0;

gen_atompos(nreps, 'Na', asize, nat, coefs, 0.0);
