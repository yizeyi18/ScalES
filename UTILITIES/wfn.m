fprintf('Default parameters for Na with 1*4*4 unit cells\n\n');

str_wfn_glb = input('global wfn file (wfn_glb.dat)','s');
if( isempty(str_wfn_glb))
  str_wfn_glb = 'wfn_glb.dat';
end

str_wfn_dg  = input('wfn dg file (wfn_dg.dat)','s');
if( isempty(str_wfn_dg))
  str_wfn_dg = 'wfn_dg.dat';
end

dim = input('Watch direction [1, 2, (3)]:\n');
if( isempty(dim) )
  dim = 3;
end

nelem = input('Number of elements along this direction (4):\n');
if( isempty(nelem) )
  nelem = 4;
end

eigind = input(strcat('Index of eigenvalue (1):\n'));
if( isempty(eigind) )
  eigind = 1;
end

% Rotate the subspace to align the degenerate eigenvectors.
subind = input(strcat('Index of eigenvalue subspaces (1):\n'));
if( isempty(subind) )
  subind = 1;
end

if( isempty(find(eigind == subind)) )
  error('eigenvalue index should be in the set of subspace index');
end

index = input(strcat('Index of the 3 dimensions for the wave function\n',...
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

fclose(fid_wfn_glb);

fid_wfn_dg = fopen(str_wfn_dg,'r');

Neigs_dg = fscanf(fid_wfn_dg, '%d', 1);
Ns_dg = fscanf(fid_wfn_dg, '%d', 3);
Ls_dg = fscanf(fid_wfn_dg, '%lf', 3);
Occ_dg = fscanf(fid_wfn_dg, '%lf', Neigs_dg);

Ns = Ns_dg;
Ntot = prod(Ns_dg);
Ls = Ls_dg;

wfn_dg = zeros(Ntot, Neigs_dg);

for i = 1 : Neigs_dg
  ind_eig = fscanf(fid_wfn_dg, '%d', 1);
  Ntot    = fscanf(fid_wfn_dg, '%d', 1);
  wfn_dg(:, i) = fscanf(fid_wfn_dg, '%lf', Ntot);
end

fclose(fid_wfn_dg);


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

% wfn_proj = wfn_dg*(wfn_dg'*wfn_glb);
% 
% wfn_glb_tmp = reshape(wfn_glb(:, eigind), Ns(1), Ns(2), Ns(3));
% wfn_dg_tmp  = reshape(wfn_proj(:, eigind),  Ns(1), Ns(2), Ns(3));

wfn_proj = wfn_dg(:,subind)*(wfn_dg(:,subind)'*wfn_glb(:,subind));
relind = find(eigind == subind);

wfn_glb_tmp = reshape(wfn_glb(:, eigind), Ns(1), Ns(2), Ns(3));
wfn_dg_tmp  = reshape(wfn_proj(:, relind),  Ns(1), Ns(2), Ns(3));

figure(1)
set(gcf, 'PaperPositionMode', 'manual');
set( gcf, 'PaperUnits', 'inch' );
set( gcf, 'PaperType', 'usletter' );
width = 3.0;
height = 3.0;
set( gcf, 'PaperPosition', [0, 0, width, height] );
set(gca,'fontsize',10);
clf

hold on
plot(x,squeeze(wfn_glb_tmp(I,J,K)),'b');
plot(x,squeeze(wfn_dg_tmp(I,J,K)),'r--');
axis tight
YLim = get(gca,'YLim');
for i = 1 : (nelem-1)
  plot(Ls(dim)*i/nelem*ones(1,2), YLim, 'k-.');
end
hold off
box on 
xlabel('x(a.u.)');
ylabel('\psi');


figure(2)
set(gcf, 'PaperPositionMode', 'manual');
set( gcf, 'PaperUnits', 'inch' );
set( gcf, 'PaperType', 'usletter' );
width = 3.0;
height = 3.0;
set( gcf, 'PaperPosition', [0, 0, width, height] );
set(gca,'fontsize',10);
clf

hold on
plot(x,squeeze(wfn_glb_tmp(I,J,K))-squeeze(wfn_dg_tmp(I,J,K)),'b-');
axis tight
YLim = get(gca,'YLim');
for i = 1 : (nelem-1)
  plot(Ls(dim)*i/nelem*ones(1,2), YLim, 'k-.');
end
hold off
box on 
xlabel('x(a.u.)');
ylabel('\psi');
