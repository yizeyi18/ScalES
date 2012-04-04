fprintf('Default parameters for Na with 1*4*4 unit cells\n\n');

str_wfn_glb = input('global wfn file (wfn_glb.dat)','s');
if( isempty(str_wfn_glb))
  str_wfn_glb = 'wfn_glb.dat';
end

dim = input('Watch direction [1, 2, (3)]:\n');
if( isempty(dim) )
  dim = 3;
end

index = input(strcat('Index of the 3 dimensions for the density matrix rho(x,:)\n',...
               '(dim dimension will be neglected) [i j k] ([4 4 36]):\n'));
if( isempty(index) )
  index = [4 4 36];
end


fid_wfn_glb = fopen(str_wfn_glb,'r');

Neigs_glb = fscanf(fid_wfn_glb, '%d', 1);
Ns_glb = fscanf(fid_wfn_glb, '%d', 3);
Ls_glb = fscanf(fid_wfn_glb, '%lf', 3);
Occ_glb = fscanf(fid_wfn_glb, '%lf', Neigs_glb);

Ns = Ns_glb;
Ntot = prod(Ns_glb);
Ls = Ls_glb;

wfn_glb = zeros(Ntot, Neigs_glb);

for i = 1 : Neigs_glb
  ind_eig = fscanf(fid_wfn_glb, '%d', 1);
  Ntot    = fscanf(fid_wfn_glb, '%d', 1);
  wfn_glb(:, i) = fscanf(fid_wfn_glb, '%lf', Ntot);
end

ind1 = index1(index, Ns);
rhomat_glb = zeros(Ntot, 1);
for i = 1 : Neigs_glb
  rhomat_glb = rhomat_glb + Occ_glb(i) * wfn_glb(ind1, i) * ...
    wfn_glb(:, i);
end

% normalize the off diagonal elements of density matrix using the same
% normalization factor as the density
rhomat_glb = (2.0 * prod(Ns) / prod(Ls)) * rhomat_glb;

rhomat_glb = reshape(rhomat_glb, [Ns(1) Ns(2) Ns(3)]);

N = Ns(dim);
x = (0:N-1)' * (Ls(dim)/N);  % mesh
I = index(1); J = index(2); K = index(3);
% I, J, K are arrays.  Only the dim dimension is expanded
switch( dim )
  case 1
    I = 1 : N;
  case 2
    J = 1 : N;
  case 3 
    K = 1 : N;
end

figure(1)
set(gcf, 'PaperPositionMode', 'manual');
set( gcf, 'PaperUnits', 'inch' );
set( gcf, 'PaperType', 'usletter' );
width = 3.0;
height = 3.0;
set( gcf, 'PaperPosition', [0, 0, width, height] );
set(gca,'fontsize',10);

plot(x,squeeze(rhomat_glb(I,J,K)),'b');
box on 
axis tight
xlabel('x(a.u.)');
ylabel('\rho(x,x\prime)');
