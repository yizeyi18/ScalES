% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xyz_to_dgdft.m 
%
% This script converts the commonly found .xyz file fromat to the 
% dgdft/pwdft input format. The cell size in Angstroms and the 
% input/output file names need to be modified suitably.
% use shiftz_flag = 1 if you are dealing with a quasi-2D system and would
% like to place the atoms at the centre of the cell (i.e., xred = 0.5)
% In order to deal with other elements, the dictionary variables 
% elem_cell_list and at_num_list need to be updated.
% 
% Amartya Banerjee, LBL, July 2017

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

% Cell dimensions in Angstroms
% Change these values as you require
acell_x_ang = 65.7756737083069; 
acell_y_ang = 43.8180033711000;
acell_z_ang = 7.9376;

% Input and output filenames
% Change these values as you require
fname_in = 'dislocation1.xyz'; 
fname_out = 'dgdft_dislocation_1.txt';

% This flag is useful for quasi-2D systems
% If this is 1, we shift z direction atoms to cell center
shiftz_flag = 0; 

% % % % % 
% Add extra elements and corresponding atomic numbers here
elem_cell_list = {'H'; 'C'; 'Na'; 'Si'; 'Al'; 'O'; 'Li'; 'F'};
at_num_list = [1 6 11 14 13 8 3 9];


% Processing starts here
angstrom_to_bohr = 1.88973;

acell_x_bohr = acell_x_ang * angstrom_to_bohr;
acell_y_bohr = acell_y_ang * angstrom_to_bohr;
acell_z_bohr = acell_z_ang * angstrom_to_bohr;

fid_in = fopen(fname_in,'r');

% Get the number of atoms
tline = fgetl(fid_in);
N_atoms = sscanf(tline, '%d');

% Get the next comment line
tline = fgetl(fid_in);
% disp(tline);

% Allocate space
atom_list = zeros(N_atoms, 4); 

% Read in the atoms
for ii=1:N_atoms
    
    tline = fgetl(fid_in);
    
    % Locate the first blank
    bl = find(tline == ' ');
    
    at_type_str = tline(1:bl(1)-1);    
    coords = tline(bl(1)+1:end);
    
    ind = find(~cellfun('isempty', strfind(elem_cell_list,at_type_str)));
    
    
    if(isempty(ind))
       fprintf(' \n Atom type %s not found !! \n', at_type_str); 
    else
        
       at_type_num = at_num_list(ind);
       atom_list(ii, 1) = at_type_num;
        
    end    
    
    at_pos = sscanf(coords,'%f');
    atom_list(ii, 2:4) = at_pos;
    
end

fclose(fid_in);

% Now write in dgdft format
sz = length(at_num_list);

fid_out = fopen(fname_out,'w');

fprintf(fid_out,'begin Super_Cell');
fprintf(fid_out,'\n %f %f %f',acell_x_bohr, acell_y_bohr, acell_z_bohr);
fprintf(fid_out,'\nend Super_Cell\n');

% Count the species
species_num = 0;
for zz = 1:sz
    atype = at_num_list(zz);
    ind = (atom_list(:,1) == atype);
    sum_ind = sum(ind,1);
    if(sum_ind > 0)
        species_num = species_num +1;
    end
end
fprintf('\n Found %d species of atoms.\n\n', species_num);

fprintf(fid_out,'\nAtom_Types_Num: %d\n', species_num);

sum_print_list = 0;
for zz = 1:sz
    
    atype = at_num_list(zz);
 
    ind = (atom_list(:,1) == atype);
    
    xlist_red = atom_list(ind,2) / acell_x_ang;
    ylist_red = atom_list(ind,3) / acell_y_ang;
    zlist_red = atom_list(ind,4) / acell_z_ang;
    
    ll = length(xlist_red);
    
    if(ll ~= 0)
     fprintf(fid_out, '\nAtom_Type: %d\n', atype);
     fprintf(fid_out, '\nbegin Atom_Red');
    end
    
    for ww = 1:ll
        
        % Adjust the reduced coordinates to be between 0 and +1.00
        if(xlist_red(ww) > 1.0)
            xlist_red(ww) = xlist_red(ww) - 1.00;
        end
        if(xlist_red(ww) < 0.0)
            xlist_red(ww) = xlist_red(ww) + 1.00;
        end
        
        if(ylist_red(ww) > 1.0)
            ylist_red(ww) = ylist_red(ww) - 1.00;
        end
        if(ylist_red(ww) < 0.0)
            ylist_red(ww) = ylist_red(ww) + 1.00;
        end
        
        if(zlist_red(ww) > 1.0)
            zlist_red(ww) = zlist_red(ww) - 1.00;
        end
        if(zlist_red(ww) < 0.0)
            zlist_red(ww) = zlist_red(ww) + 1.00;
        end
        
        if(shiftz_flag == 0)
              
          fprintf(fid_out,'\n %f %f %f', xlist_red(ww), ylist_red(ww), ...
                                    zlist_red(ww));
        else
          fprintf(fid_out,'\n %f %f %f', xlist_red(ww), ylist_red(ww), ...
                                    0.5);           
        end
    end
    
    if(ll ~= 0)
     fprintf(fid_out, '\nend Atom_Red\n');
    end
    
    sum_print_list = sum_print_list + ll;
    fprintf('\n Printed %d atoms with At. Num. %d', ll, atype);
end

fclose(fid_out);

fprintf('\n\n %d Total atom details printed.\n', sum_print_list);
fprintf('\n\n Note: shiftz value = %d. \n\n', shiftz_flag);