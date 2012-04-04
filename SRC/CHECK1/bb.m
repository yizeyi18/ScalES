%-----------------------------------------------------------
if(1)
    %testing for -1/2 Laplacian
    
    %specify dimension of the problem
    N1 = 4;  N2 = 4;  N3 = 4;
    L1 = 1;    L2 = 1;    L3 = 1;
    h1 = L1/N1;  h2 = L2/N2;  h3 = L3/N3;
    
    Ndeg = 3; %polynomials basis is used now, max deg
    
    Nlbl1 = 5; %LBL grid
    Nlbl2 = 5;
    Nlbl3 = 5;
    
    alpha = 20;
    beta = 0.5;
    
    Nele = N1*N2*N3;
    Neigperele = 1; %number of eigpars per element
    Neigttl = Neigperele*Nele; %number of total eigpairs wanted
    Norbperele = 7; %number of orbitals per element
    EPS = 0; %SVD CUTOFF VALUE
    
    delta = 0.1; %control the width of the weight wall
    bc = 'p';
end

%-----------------------------------------------------------
if(1)
    %construct local basis (poly for now)
    basis_all = cell(N1,N2,N3);  %Nbasis_per_cell = zeros(N1,N2,N3);
    
    [x1,w1,P1,D1,S1,T1] = lgl(Nlbl1-1);x1 = x1/2*h1;w1 = w1/2*h1;D1 = D1*2/h1;
    [x2,w2,P2,D2,S2,T2] = lgl(Nlbl2-1);x2 = x2/2*h2;w2 = w2/2*h2;D2 = D2*2/h2;
    [x3,w3,P3,D3,S3,T3] = lgl(Nlbl3-1);x3 = x3/2*h3;w3 = w3/2*h3;D3 = D3*2/h3;
    
    [gg1,gg2,gg3] = ndgrid(x1, x2, x3);
    www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]); %weight within each cell
    
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                
                if(i1==1 && i2==1 && i3==1)
                %center
                xx1 = gg1;        xx2 = gg2;        xx3 = gg3;        ee = ones(size(xx1));
                cnt = 0;
                VL=1;        DX=2;        DY=3;        DZ=4;
                basis = cell(1000,1);
                for d1=0:Ndeg
                    for d2=0:Ndeg
                        for d3=0:Ndeg
                            if(d1+d2+d3<=Ndeg)
                                cnt = cnt+1;
                                basis{cnt} = xx1.^d1 .* xx2.^d2 .* xx3.^d3;
                            end
                        end
                    end
                end
                basis = basis(1:cnt);
                if(1)
                    VLtmp = zeros(Nlbl1*Nlbl2*Nlbl3, cnt);
                    for g=1:cnt
                        VLtmp(:,g) = basis{g}(:);
                    end
                    tmp = VLtmp;
                    for g=1:cnt
                        tmp(:,g) = tmp(:,g) .* sqrt(www(:));
                    end
                    [U,S,V] = svd(tmp,0);
                    G = V*inv(S); %transformation matrix
                    VLtmp = VLtmp * G;
                    for g=1:cnt
                        basis{g} = reshape(VLtmp(:,g), [Nlbl1,Nlbl2,Nlbl3]);
                    end
                end
                end
                
                basis_all{i1,i2,i3} = basis;
            end
        end
    end
    
    Ncells = N1*N2*N3;
end

%-----------------------------------------------------------
if(1)
    %generate potential
    tx1 = [];        for g1=1:N1            tx1 = [tx1; x1+(g1-1/2)*h1];        end
    tx2 = [];        for g2=1:N2            tx2 = [tx2; x2+(g2-1/2)*h2];        end
    tx3 = [];        for g3=1:N3            tx3 = [tx3; x3+(g3-1/2)*h3];        end
    [ox1,ox2,ox3] = ndgrid(tx1,tx2,tx3); %location
    vot = zeros(size(gx1));
    for g1=1:N1
        for g2=1:N2
            for g3=1:N3
                c1 = (g1-1/2)*h1;                c2 = (g2-1/2)*h2;                c3 = (g3-1/2)*h3;
                c1 = c1 + 0.2*(rand(1)-1/2)*h1;                c2 = c2 + 0.2*(rand(1)-1/2)*h2;                c3 = c3 + 0.2*(rand(1)-1/2)*h3;
                s1 = h1/2;                            s2 = h2/2;                            s3 = h3/2;
                gx1 = ox1-c1;                bad = find(gx1>1/2); gx1(bad) = gx1(bad)-1;                bad = find(gx1<-1/2); gx1(bad) = gx1(bad)+1;
                gx2 = ox2-c2;                bad = find(gx2>1/2); gx2(bad) = gx2(bad)-1;                bad = find(gx2<-1/2); gx2(bad) = gx2(bad)+1;
                gx3 = ox3-c3;                bad = find(gx3>1/2); gx3(bad) = gx3(bad)-1;                bad = find(gx3<-1/2); gx3(bad) = gx3(bad)+1;
                vot = vot - exp(-((gx1/s1).^2+(gx2/s2).^2+(gx3/s3).^2));
            end
        end
    end
    tmp = vot;
    vot = cell(N1,N2,N3);
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                aux = tmp([1:Nlbl1]+(i1-1)*Nlbl1, [1:Nlbl2]+(i2-1)*Nlbl2, [1:Nlbl3]+(i3-1)*Nlbl3);
                vot{i1,i2,i3} = aux;
            end
        end
    end
    %imagesc(tx1,tx2,tmp(:,:,10));    error('here');
end

%-----------------------------------------------------------
if(0)
    %assemble DG and solve for eigvals
    [A, M, INDEX] = setupDG(N1,N2,N3, L1,L2,L3, Nlbl1,Nlbl2,Nlbl3, alpha,beta, basis_all, vot);
    Nbas = size(A,1);
    [V, E] = eig(A);    E = diag(E); %V is always orthogonal
    E = sort(real(E));
    fprintf(1, 'eigvals from adaptive basis functions\n');
    ext = E(1:Neigttl) / (beta*4*pi^2)
end

%-----------------------------------------------------------
if(1)
    %calculate the weight matrix
    weight = cell(3,3,3);
    for g1=1:3
        for g2=1:3
            for g3=1:3
                tx1 = x1 + (g1-2)*h1;
                tx2 = x2 + (g2-2)*h2;
                tx3 = x3 + (g3-2)*h3;
                [gg1,gg2,gg3] = ndgrid(tx1,tx2,tx3);
                gg = sqrt(gg1.^2+gg2.^2+gg3.^2);
                %tmp = (abs(gg1)<(1-delta)*h1*1.5) & (abs(gg2)<(1-delta)*h2*1.5) & (abs(gg3)<(1-delta)*h3*1.5);  weight{g1,g2,g3} = 1-tmp;
                weight{g1,g2,g3} = (gg>(1-delta)*h1*1.5);
            end
        end
    end
    
    %for each element, form 3 by 3 DG problem (with PBC?), find the low eig, do local opt along with ngbd low eigvecs
    %C is the contraction matrix
    Atall = cell(N1,N2,N3);
    C = zeros(Nbas, Nele * Norbperele);
    cnt = 0;
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                aux1 = mod([i1-1:i1+1]+N1-1,N1)+1;
                aux2 = mod([i2-1:i2+1]+N2-1,N2)+1;
                aux3 = mod([i3-1:i3+1]+N3-1,N3)+1;
                
                %form 3 by 3 problem, get the low eigval modes, put into Ct
                basis_aux = cell(3,3,3);
                vot_aux = cell(3,3,3);
                map_aux = [];
                for g1=1:3
                    for g2=1:3
                        for g3=1:3
                            basis_aux{g1,g2,g3} = basis_all{aux1(g1), aux2(g2), aux3(g3)};
                            vot_aux{g1,g2,g3} = vot{aux1(g1), aux2(g2), aux3(g3)};
                            map_aux = [map_aux, INDEX{aux1(g1),aux2(g2),aux3(g3)}];
                        end
                    end
                end
                
                %if(i1==1 && i2==1 && i3==1)
                
                if(bc=='p')
                    [At, Mt, INDEXt] = setupDG(3,3,3, h1*3,h2*3,h3*3, Nlbl1,Nlbl2,Nlbl3, alpha,beta, basis_aux, vot_aux);
                else
                    At = A(map_aux,map_aux); %dirichlet
                end
                Atall{i1,i2,i3} = At;
                [Vt, Et] = eig(At);                %tmp = sort(diag(Et));                tmp(1:10)' / (beta*(2*pi/(h1*3))^2)
                
                %Ct = Vt(:,1:(3*3*3 *Norbperele));
                Ct = Vt(:,1:(3*3*3 *2));
                %compute weight matrix Wt
                Wt = zeros(size(At));
                sss = 0;
                for g1=1:3
                    for g2=1:3
                        for g3=1:3
                            basis = basis_aux{g1,g2,g3};
                            nbasis = size(basis,1);
                            S = zeros(nbasis,nbasis);
                            for a=1:nbasis
                                for b=1:nbasis
                                    tmp = (basis{a}.*basis{b}.*weight{g1,g2,g3}).*www;
                                    S(a,b) = sum(tmp(:));
                                end
                            end
                            Wt(sss+[1:nbasis],sss+[1:nbasis]) = S;
                            sss = sss + nbasis;
                        end
                    end
                end
                %extract the ones that have small weights, put into Gt
                CVWVC = Ct'*Wt*Ct;
                CVWVC = (CVWVC + CVWVC')/2;
                [Gt, Ft] = eig(CVWVC);
                Gt = Gt(:,1:Norbperele);
                
                %end
                
                %get the matrix that map global basis to buffer basis, put into St
                nb = numel(map_aux);
                St = sparse(map_aux, 1:nb, ones(1,nb), Nbas, nb);
                %get the full transformation matrix Tt = St * Ct * Gt;
                Tt = St*(Ct*Gt);
                %tmp = zeros(size(Ct*Gt));                tmp(1:20:end) = 1;                Tt = St * tmp;
                %put Tt matrix into the global mixing matrix C
                C(:, cnt+[1:Norbperele]) = Tt;
                cnt = cnt+Norbperele;
            end
        end
    end
    
end




%-----------------------------------------------------------
if(1)
    Ac = C'*A*C;
    Sc = C'*C;
    Ac = (Ac+Ac')/2;
    Sc = (Sc+Sc')/2;
    [Vc, Ec] = eig(Ac, Sc);    Ec = diag(Ec);
    Ec = sort(real(Ec));
    fprintf(1, 'eigvals from locally contracted adaptive basis functions\n');
    if(bc=='p')
        per = Ec(1:Neigttl) / (beta*4*pi^2)
    else
        drt = Ec(1:Neigttl) / (beta*4*pi^2)
    end
end

if(1)
    [ext per]
    cond(C)
end







