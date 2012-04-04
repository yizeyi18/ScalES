fprintf('Default parameters for Na with 1*4*4 unit cells\n\n');

str_buf = input('buffer vtot file (vtot)','s');
if( isempty(str_buf))
  str_buf = 'vtot';
end

dim = input('Watch direction [1, 2, (3)]:\n');
if( isempty(dim) )
  dim = 3;
end


index = input(strcat('Index of the location to watch the wave function\n',...
               '(dim dimension will be replaced by all the points along that direction) [i j k] ([5 13 13]):\n'));
if( isempty(index) )
  index = [5 13 13];
end


fid_buf = fopen(str_buf ,'r');

Ns = fscanf(fid_buf, '%d', 3);
Ls = fscanf(fid_buf, '%lf', 3);

Ntot = prod(Ns);

vtot = zeros(Ntot, 1);
vtot = fscanf(fid_buf, '%lf', Ntot);

fclose(fid_buf);


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
clf

box on 
vtottmp = reshape(vtot, Ns(1), Ns(2), Ns(3));
surf(squeeze(vtottmp(I,:,:)));
% plot(x,squeeze(vtottmp(I,J,K)),'b');
axis tight
xlabel('x(a.u.)');
ylabel('\psi');


