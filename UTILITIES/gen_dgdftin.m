function gen_dgdftin(nreps, nelems, bufferratio, atype, ...
  asize, nat, coefs, ns_glb, ns_elem, ran, mode);

nrepx = nreps(1); nrepy = nreps(2); nrepz = nreps(3);
nelemx = nelems(1); nelemy = nelems(2); nelemz = nelems(3);
nelem = prod(nelems);
bufferratiox = bufferratio(1);
bufferratioy = bufferratio(2);
bufferratioz = bufferratio(3);

fprintf('Constructing %5s cells with size %3d*%3d*%3d\n', atype, ...
  nreps(1), nreps(2), nreps(3));
fprintf('Total number of atoms:   %6d\n\n', nat * prod(nreps));


% Molecule Level 1
fprintf('Level 1...\n');

if( numel(asize) == 1 )
  asize = ones(3,1)*asize;
elseif( numel(asize) == 3 )
  % do nothing
else
  error('asize is not in the correct form');
end
C = diag(asize);

xyzmat = coefs*C';
%
% repeat the cell nrep times along the z direction
%
for krep = 1 : nrepz
  for jrep = 1 : nrepy
    for irep = 1 : nrepx
      if( irep ~= 1 || jrep ~= 1 || krep ~= 1 ) 
	xyzpick = xyzmat(1:nat,:);
	xyzpick(:,1) = xyzpick(:,1) + (irep-1)*asize(1);
	xyzpick(:,2) = xyzpick(:,2) + (jrep-1)*asize(2);
	xyzpick(:,3) = xyzpick(:,3) + (krep-1)*asize(3);
	xyzmat = [xyzmat; xyzpick];
      end
    end
  end
end
% Add some randomness
xyzmat = xyzmat + ran * (randn(size(xyzmat)));
%
% modify the supercell
%
C(1,1) = nrepx*asize(1);
C(2,2) = nrepy*asize(2);
C(3,3) = nrepz*asize(3);
%
% 4. Configure the molecule (crystal)
%

posstart = [0;0;0];
posidx = [0;0;0];


% Redefine the grid points
n11 = ns_glb(1) * nrepx;
n21 = ns_glb(2) * nrepy;
n31 = ns_glb(3) * nrepz;
n11 = round(n11 / nelemx)*nelemx;
n21 = round(n21 / nelemy)*nelemy;
n31 = round(n31 / nelemz)*nelemz;

L11 = C(1,1);
L21 = C(2,2);
L31 = C(3,3);
h11 = L11/n11;
h21 = L21/n21;
h31 = L31/n31;


fh = fopen('dgdft','w');

fprintf(fh, 'Mixing_Alpha: \t %4.2f\n\n', 0.5);
fprintf(fh, 'Solver: \t %7s\n', mode);
fprintf(fh, 'Mapping_Mode: \t uniform\n\n');
fprintf(fh, 'Extra_States: \t %4d\n\n', 10);

fprintf(fh, 'Temperature: \t %4d\n', 300);
fprintf(fh, 'SCF_Maxiter: \t %4d\n', 20);
fprintf(fh, 'Eig_Maxiter: \t %4d\n\n', 20);

fprintf(fh, 'PeriodTable_Dir = ../\n\n');

fprintf(fh, 'Enrich_Number: \t %4d\n', 6);
fprintf(fh, 'DG_Degree: \t %4d\n', 0);
fprintf(fh, 'DG_Alpha: \t %4d\n\n', 20);

fprintf(fh, 'MB: \t %4d\n\n', 16);

fprintf(fh, 'Output_Dir = ./\n');
fprintf(fh, 'Output_Density = %4d\n', 0);
fprintf(fh, 'Output_Wfn = %4d\n\n', 0);

fprintf(fh, '#Restart_Mode : restart\n');
fprintf(fh, 'Restart_Mode : from_scratch\n');
fprintf(fh, 'Restart_Density : rho_glb.dat\n\n');

  
fprintf(fh, '# Molecule Global\n\n');

fprintf(fh, 'begin Super_Cell\n', 1);
fprintf(fh, '%12.6f     %12.6f    %12.6f\n', C(1,1), C(2,2), C(3,3));
fprintf(fh, 'end Super_Cell\n\n', 1);

fprintf(fh, 'begin Grid_Size\n', 1);
fprintf(fh, '%6d       %6d      %6d\n', n11, n21, n31);
fprintf(fh, 'end Grid_Size\n\n', 1);

fprintf(fh, 'Atom_Types_Num:   %6d\n\n', 1);
fprintf(fh, 'Atom_Type:        %6s\n\n', atype);

fprintf(fh, 'begin Atom_Coord\n', 1);
fprintf(fh, '%12.6f     %12.6f    %12.6f\n', xyzmat');
fprintf(fh, 'end Atom_Coord\n\n', 1);

fprintf(fh, '# DG USE \n\n');

fprintf(fh, 'begin Element_Size\n', 1);
fprintf(fh, '%6d       %6d      %6d\n', nelemx, nelemy, nelemz);
fprintf(fh, 'end Element_Size\n\n', 1);


%% Molecule Level 2
fprintf('Level 2...\n');

for iz = 1 : nelemz
  for iy = 1 : nelemy
    for ix = 1 : nelemx
      ie = ix + (iy-1)*nelemx + (iz-1)*nelemx*nelemy;

      % Buffer extends by half unit cell along both directions in z.

      posstart = [ -bufferratiox+(ix-1) * (nrepx/nelemx); 
		   -bufferratioy+(iy-1) * (nrepy/nelemy); 
		   -bufferratioz+(iz-1) * (nrepz/nelemz)];
      posstart(1) = posstart(1) * asize(1);
      posstart(2) = posstart(2) * asize(2);
      posstart(3) = posstart(3) * asize(3);


      posidx = zeros(3,1);
      posidx = round(posstart./[h11;h21;h31]); 
      posstart = posidx .* [h11;h21;h31]; 

      C(1,1) = (nrepx / nelemx + bufferratiox*2) * asize(1);
      C(2,2) = (nrepy / nelemy + bufferratioy*2) * asize(2);
      C(3,3) = (nrepz / nelemz + bufferratioz*2) * asize(3);  

      C(1,1) = round(C(1,1) / h11)*h11;
      C(2,2) = round(C(2,2) / h21)*h21;
      C(3,3) = round(C(3,3) / h31)*h31;

      n12 = round(C(1,1)/h11);
      n22 = round(C(2,2)/h21);
      n32 = round(C(3,3)/h31);


      fprintf(fh, '# Buffer %6d\n\n', ie);
      fprintf(fh, 'begin Buffer_Position_Start\n', 1);
      fprintf(fh, '%12.6f     %12.6f    %12.6f\n', posstart(1), posstart(2), posstart(3));
      fprintf(fh, 'end Buffer_Position_Start\n\n', 1);

      fprintf(fh, 'begin Buffer_Cell\n', 1);
      fprintf(fh, '%12.6f     %12.6f    %12.6f\n', C(1,1), C(2,2), C(3,3));
      fprintf(fh, 'end Buffer_Cell\n\n', 1);

      fprintf(fh, 'begin Buffer_Grid_Size\n', 1);
      fprintf(fh, '%6d       %6d      %6d\n', n12, n22, n32);
      fprintf(fh, 'end Buffer_Grid_Size\n\n', 1);

    end
  end
end


%% Molecule Level 3
fprintf('Level 3...\n');

for iz = 1 : nelemz
  for iy = 1 : nelemy
    for ix = 1 : nelemx
      ie = ix + (iy-1)*nelemx + (iz-1)*nelemx*nelemy;

      posstart = [ (ix-1) * (nrepx/nelemx); 
		   (iy-1) * (nrepy/nelemy); 
                   (iz-1) * (nrepz/nelemz)];
      posstart(1) = posstart(1) * asize(1);
      posstart(2) = posstart(2) * asize(2);
      posstart(3) = posstart(3) * asize(3);

      posidx = zeros(3,1);
      posidx = round(posstart./[h11;h21;h31]); 
      posstart = posidx .* [h11;h21;h31]; 

      C(1,1) = (nrepx / nelemx) * asize(1);  
      C(2,2) = (nrepy / nelemy) * asize(2);  
      C(3,3) = (nrepz / nelemz) * asize(3);  

      C(1,1) = round(C(1,1) / h11)*h11;
      C(2,2) = round(C(2,2) / h21)*h21;
      C(3,3) = round(C(3,3) / h31)*h31;

      fprintf(fh, '# Element %6d\n\n', ie);
      fprintf(fh, 'begin Element_Position_Start\n', 1);
      fprintf(fh, '%12.6f     %12.6f    %12.6f\n', posstart(1), posstart(2), posstart(3));
      fprintf(fh, 'end Element_Position_Start\n\n', 1);

      fprintf(fh, 'begin Element_Cell\n', 1);
      fprintf(fh, '%12.6f     %12.6f    %12.6f\n', C(1,1), C(2,2), C(3,3));
      fprintf(fh, 'end Element_Cell\n\n', 1);

      fprintf(fh, 'begin Element_Grid_Size\n', 1);
      fprintf(fh, '%6d       %6d      %6d\n', ns_elem(1), ns_elem(2), ns_elem(3));
      fprintf(fh, 'end Element_Grid_Size\n\n', 1);

    end
  end
end


fclose(fh);

fprintf('\n Done. input file generated in dgdft.\n');
