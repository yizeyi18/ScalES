fprintf('Default parameters for Na with 1*4*4 unit cells\n\n');

str_rho_glb = input('global rho file (rho_glb.dat)','s');
if( isempty(str_rho_glb))
  str_rho_glb = 'rho_glb.dat';
end

str_rho_dg = input('dg rho file (rho_dg.dat)','s');
if( isempty(str_rho_dg) )
  str_rho_dg = 'rho_dg.dat';
end

dim = input('Watch direction [1, 2, (3)]:\n');
if( isempty(dim) )
  dim = 3;
end

nelem = input('Number of elements along this direction (4):\n');
if( isempty(nelem) )
  nelem = 4;
end

index = input('Index of the 3 dimensions (dim dimension will be neglected) [i j k] ([4 4 4]):\n');
if( isempty(index) )
  index = [4 4 4];
end


fid_rho_glb = fopen(str_rho_glb,'r');
fid_rho_dg  = fopen(str_rho_dg,'r');

Ns_glb = fscanf(fid_rho_glb, '%d', 3);
Ls_glb = fscanf(fid_rho_glb, '%lf', 3);
Ntot_glb = fscanf(fid_rho_glb, '%d', 1);
if( prod(Ns_glb) ~= Ntot_glb )
  error('Ns does not match');
end

Ns_dg  = fscanf(fid_rho_dg, '%d', 3);
Ls_dg  = fscanf(fid_rho_dg, '%lf', 3);
Ntot_dg = fscanf(fid_rho_dg, '%d', 1);
if( prod(Ns_dg) ~= Ntot_dg)
  error('Ns does not match');
end

if( Ntot_glb ~= Ntot_dg)
  error('Ntot does not match');
end

Ns = Ns_glb;
Ls = Ls_glb;
Ntot = Ntot_glb;

rho_glb = fscanf(fid_rho_glb, '%lf', inf);
rho_dg  = fscanf(fid_rho_dg, '%lf', inf);

rho_glb = reshape(rho_glb, Ns(1), Ns(2), Ns(3));
rho_dg  = reshape(rho_dg,  Ns(1), Ns(2), Ns(3));

fclose(fid_rho_glb);
fclose(fid_rho_dg);

N = size(rho_glb, dim);
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
clf

hold on
plot(x,squeeze(rho_glb(I,J,K)),'b',x,squeeze(rho_dg(I,J,K)),'r--');
axis tight
YLim = get(gca,'YLim');
for i = 1 : (nelem-1)
  plot(Ls(dim)*i/nelem*ones(1,2), YLim, 'k-.');
end
hold off

box on 
xlabel('x(a.u.)');
ylabel('\rho');
legend('Global', 'DG');

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
plot(x,squeeze(rho_glb(I,J,K))-squeeze(rho_dg(I,J,K)),'r');
axis tight
YLim = get(gca,'YLim');
for i = 1 : (nelem-1)
  plot(Ls(dim)*i/nelem*ones(1,2), YLim, 'k-.');
end
hold off
box on 
xlabel('x(a.u.)');
ylabel('\Delta\rho');

% figure(3)
% imagesc(squeeze(rho_glb(:,:,4))-squeeze(rho_dg(:,:,4)))
% colorbar;
