function [A, M, INDEX] = setupDG(N1,N2,N3, L1,L2,L3, Nlbl1,Nlbl2,Nlbl3, alpha, beta, basis_all, vot)

h1 = L1/N1;  h2 = L2/N2;  h3 = L3/N3;

%form INDEX
Ncells = N1*N2*N3;
INDEX = cell(N1,N2,N3);
cnt = 0;
for i1=1:N1
    for i2=1:N2
        for i3=1:N3
            ntmp = size(basis_all{i1,i2,i3},1);
            INDEX{i1,i2,i3} = cnt+[1:ntmp];
            cnt = cnt + ntmp;
        end
    end
end
Ndof = cnt;

[x1,w1,P1,D1,S1,T1] = lgl(Nlbl1-1);x1 = x1/2*h1;w1 = w1/2*h1;D1 = D1*2/h1;
[x2,w2,P2,D2,S2,T2] = lgl(Nlbl2-1);x2 = x2/2*h2;w2 = w2/2*h2;D2 = D2*2/h2;
[x3,w3,P3,D3,S3,T3] = lgl(Nlbl3-1);x3 = x3/2*h3;w3 = w3/2*h3;D3 = D3*2/h3;

[gg1,gg2,gg3] = ndgrid(x1, x2, x3);
www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]); %weight within each cell

%form derivatives %basis_all from (i1,i2,i3) (g) (:,:,:) to (i1,i2,i3) (g,VXYZ) (:,:,:)
if(1)
    basis_old = basis_all;
    basis_all = cell(N1,N2,N3);
    VL=1;        DX=2;        DY=3;        DZ=4;
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                bo = basis_old{i1,i2,i3};
                nbasis = size(bo,1);
                bn = cell(nbasis,4);
                for g=1:nbasis
                    tmpval = bo{g};
                    %
                    aux = permute(tmpval,[1,2,3]);
                    new = reshape(aux,[size(aux,1),size(aux,2)*size(aux,3)]);
                    new = D1*new;aux = reshape(new,size(aux));
                    tmpdx = permute(aux,[1,2,3]);
                    %
                    aux = permute(tmpval,[2,1,3]);
                    new = reshape(aux,[size(aux,1),size(aux,2)*size(aux,3)]);
                    new = D2*new;aux = reshape(new,size(aux));
                    tmpdy = permute(aux,[2,1,3]);
                    %
                    aux = permute(tmpval,[3,2,1]);
                    new = reshape(aux,[size(aux,1),size(aux,2)*size(aux,3)]);
                    new = D3*new;aux = reshape(new,size(aux));
                    tmpdz = permute(aux,[3,2,1]);
                    %
                    bn{g,VL} = tmpval;
                    bn{g,DX} = tmpdx;
                    bn{g,DY} = tmpdy;
                    bn{g,DZ} = tmpdz;
                end
                basis_all{i1,i2,i3} = bn;
            end
        end
    end
end

%mass matrix
if(1)
    M = zeros(Ndof,Ndof);
    %NMAX = 100*1024^2;    Iall = zeros(NMAX,1);    Jall = zeros(NMAX,1);    Sall = zeros(NMAX,1);    cnt = 0;
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                basis = basis_all{i1,i2,i3};
                nbasis = size(basis,1);
                VL=1;        DX=2;        DY=3;        DZ=4;
                S = zeros(nbasis,nbasis);
                for a=1:nbasis
                    for b=1:nbasis
                        tmp = (basis{a,VL}.*basis{b,VL}).*www;
                        S(a,b) = sum(tmp(:));
                    end
                end
                uindex = INDEX{i1,i2,i3};
                vindex = INDEX{i1,i2,i3};
                %
                M(uindex,vindex) = M(uindex,vindex) + S;
                %[I,J] = ndgrid(uindex,vindex); nbtmp = numel(I); Iall(cnt+[1:nbtmp]) = I(:); Jall(cnt+[1:nbtmp]) = J(:); Sall(cnt+[1:nbtmp]) = S(:); cnt = cnt+nbtmp;
            end
        end
    end
    %Iall = Iall(1:cnt);    Jall = Jall(1:cnt);    Sall = Sall(1:cnt);    MS = sparse(Iall,Jall,Sall);
end

%stiffness matrix
if(1)
    A = zeros(Ndof,Ndof);
    %NMAX = 100*1024^2;    Iall = zeros(NMAX,1);    Jall = zeros(NMAX,1);    Sall = zeros(NMAX,1);    cnt = 0;
    
    %term 1
    %iterate through cells
    www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]);
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                %only self-self interaction
                basis = basis_all{i1,i2,i3};
                nbasis = size(basis,1);
                S = zeros(nbasis,nbasis);
                for a=1:nbasis
                    for b=1:nbasis
                        tmp = beta*(basis{a,DX}.*basis{b,DX}+basis{a,DY}.*basis{b,DY}+basis{a,DZ}.*basis{b,DZ}).*www;
                        S(a,b) = sum(tmp(:));
                    end
                end
                uindex = INDEX{i1,i2,i3};
                vindex = INDEX{i1,i2,i3};
                %full
                A(uindex,vindex) = A(uindex,vindex) + S;
                %[I,J] = ndgrid(uindex,vindex); nbtmp = numel(I); Iall(cnt+[1:nbtmp]) = I(:); Jall(cnt+[1:nbtmp]) = J(:); Sall(cnt+[1:nbtmp]) = S(:); cnt = cnt+nbtmp;
            end
        end
    end
    
    %potential
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                %only self-self interaction
                basis = basis_all{i1,i2,i3};
                nbasis = size(basis,1);
                S = zeros(nbasis,nbasis);
                vvv = vot{i1,i2,i3};
                for a=1:nbasis
                    for b=1:nbasis
                        tmp = basis{a,VL}.*basis{b,VL}.*vvv.*www;
                        S(a,b) = sum(tmp(:));
                    end
                end
                uindex = INDEX{i1,i2,i3};
                vindex = INDEX{i1,i2,i3};
                %full
                A(uindex,vindex) = A(uindex,vindex) + S;
                %[I,J] = ndgrid(uindex,vindex); nbtmp = numel(I); Iall(cnt+[1:nbtmp]) = I(:); Jall(cnt+[1:nbtmp]) = J(:); Sall(cnt+[1:nbtmp]) = S(:); cnt = cnt+nbtmp;
            end
        end
    end
    
    
    %term 234
    %yz faces
    ww = w2*w3'; %weight within each cell
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                if i1==1
                    p1=N1;
                else
                    p1=i1-1;
                end
                %u target, v source
                VL=1;        DX=2;        DY=3;        DZ=4;
                %consider (u,v)
                for uch=[0 1]
                    for vch=[0 1]
                        if(uch==0)
                            u1=p1;              u2=i2;              u3=i3;
                        else
                            u1=i1;              u2=i2;              u3=i3;
                        end
                        basis_u = basis_all{u1,u2,u3};
                        uindex = INDEX{u1,u2,u3};
                        nbasis_u = size(basis_u,1);
                        if(vch==0)
                            v1=p1;              v2=i2;              v3=i3;
                        else
                            v1=i1;              v2=i2;              v3=i3;
                        end
                        basis_v = basis_all{v1,v2,v3};
                        vindex = INDEX{v1,v2,v3};
                        nbasis_v = size(basis_v,1);
                        %
                        S = zeros(nbasis_u,nbasis_v);
                        uDXave = cell(nbasis_u,1);
                        uVLjmp = cell(nbasis_u,1);
                        for a=1:nbasis_u
                            if(uch==0)
                                uDXave{a} = squeeze((basis_u{a,DX}(end,:,:) + 0)/2);
                                uVLjmp{a} = squeeze((basis_u{a,VL}(end,:,:)*1 + 0));
                            else
                                uDXave{a} = squeeze((basis_u{a,DX}(1,:,:) + 0)/2);
                                uVLjmp{a} = squeeze((basis_u{a,VL}(1,:,:)*-1 + 0));
                            end
                        end
                        vDXave = cell(nbasis_v,1);
                        vVLjmp = cell(nbasis_v,1);
                        for b=1:nbasis_v
                            if(vch==0)
                                vDXave{b} = squeeze((basis_v{b,DX}(end,:,:) + 0)/2);
                                vVLjmp{b} = squeeze((basis_v{b,VL}(end,:,:)*1 + 0));
                            else
                                vDXave{b} = squeeze((basis_v{b,DX}(1,:,:) + 0)/2);
                                vVLjmp{b} = squeeze((basis_v{b,VL}(1,:,:)*-1 + 0));
                            end
                        end
                        for a=1:nbasis_u
                            for b=1:nbasis_v
                                tmp1 = beta*-uDXave{a}.*vVLjmp{b}.*ww;            tmp1 = sum(tmp1(:));
                                tmp2 = beta*-vDXave{b}.*uVLjmp{a}.*ww;            tmp2 = sum(tmp2(:));
                                tmp3 = alpha/h1*(uVLjmp{a}.*vVLjmp{b}.*ww);            tmp3 = sum(tmp3(:));
                                S(a,b) = tmp1 + tmp2 + tmp3;
                            end
                        end
                        %full
                        A(uindex,vindex) = A(uindex,vindex) + S;
                        %[I,J] = ndgrid(uindex,vindex); nbtmp = numel(I); Iall(cnt+[1:nbtmp]) = I(:); Jall(cnt+[1:nbtmp]) = J(:); Sall(cnt+[1:nbtmp]) = S(:); cnt = cnt+nbtmp;
                    end
                end
                clear uDXave uVLjmp vDXave vVLjmp;
            end
        end
    end
    %norm(A(:))
    
    %zx faces
    ww = w1*w3'; %weight within each cell
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                if i2==1
                    p2=N2;
                else
                    p2=i2-1;
                end
                VL=1;        DX=2;        DY=3;        DZ=4;
                for uch=[0 1]
                    for vch=[0 1]
                        if(uch==0)
                            u1=i1;              u2=p2;              u3=i3;
                        else
                            u1=i1;              u2=i2;              u3=i3;
                        end
                        basis_u = basis_all{u1,u2,u3};
                        uindex = INDEX{u1,u2,u3};
                        nbasis_u = size(basis_u,1);
                        if(vch==0)
                            v1=i1;              v2=p2;              v3=i3;
                        else
                            v1=i1;              v2=i2;              v3=i3;
                        end
                        basis_v = basis_all{v1,v2,v3};
                        vindex = INDEX{v1,v2,v3};
                        nbasis_v = size(basis_v,1);
                        %
                        S = zeros(nbasis_u,nbasis_v);
                        uDYave = cell(nbasis_u,1);
                        uVLjmp = cell(nbasis_u,1);
                        for a=1:nbasis_u
                            if(uch==0)
                                uDYave{a} = squeeze((basis_u{a,DY}(:,end,:) + 0)/2);
                                uVLjmp{a} = squeeze((basis_u{a,VL}(:,end,:)*1 + 0));
                            else
                                uDYave{a} = squeeze((basis_u{a,DY}(:,1,:) + 0)/2);
                                uVLjmp{a} = squeeze((basis_u{a,VL}(:,1,:)*-1 + 0));
                            end
                        end
                        vDYave = cell(nbasis_v,1);
                        vVLjmp = cell(nbasis_v,1);
                        for b=1:nbasis_v
                            if(vch==0)
                                vDYave{b} = squeeze((basis_v{b,DY}(:,end,:) + 0)/2);
                                vVLjmp{b} = squeeze((basis_v{b,VL}(:,end,:)*1 + 0));
                            else
                                vDYave{b} = squeeze((basis_v{b,DY}(:,1,:) + 0)/2);
                                vVLjmp{b} = squeeze((basis_v{b,VL}(:,1,:)*-1 + 0));
                            end
                        end
                        for a=1:nbasis_u
                            for b=1:nbasis_v
                                tmp1 = beta*-uDYave{a}.*vVLjmp{b}.*ww;            tmp1 = sum(tmp1(:));
                                tmp2 = beta*-vDYave{b}.*uVLjmp{a}.*ww;            tmp2 = sum(tmp2(:));
                                tmp3 = alpha/h2*(uVLjmp{a}.*vVLjmp{b}.*ww);            tmp3 = sum(tmp3(:));
                                S(a,b) = tmp1 + tmp2 + tmp3;
                            end
                        end
                        %full
                        A(uindex,vindex) = A(uindex,vindex) + S;
                        %[I,J] = ndgrid(uindex,vindex); nbtmp = numel(I); Iall(cnt+[1:nbtmp]) = I(:); Jall(cnt+[1:nbtmp]) = J(:); Sall(cnt+[1:nbtmp]) = S(:); cnt = cnt+nbtmp;
                    end
                end
                clear uDYave uVLjmp vDYave vVLjmp;
            end
        end
    end
    %norm(A(:))
    
    %xy faces
    ww = w1*w2'; %weight within each cell
    for i1=1:N1
        for i2=1:N2
            for i3=1:N3
                if i3==1
                    p3=N3;
                else
                    p3=i3-1;
                end
                VL=1;        DX=2;        DY=3;        DZ=4;
                for uch=[0 1]
                    for vch=[0 1]
                        if(uch==0)
                            u1=i1;              u2=i2;              u3=p3;
                        else
                            u1=i1;              u2=i2;              u3=i3;
                        end
                        basis_u = basis_all{u1,u2,u3};
                        uindex = INDEX{u1,u2,u3};
                        nbasis_u = size(basis_u,1);
                        if(vch==0)
                            v1=i1;              v2=i2;              v3=p3;
                        else
                            v1=i1;              v2=i2;              v3=i3;
                        end
                        basis_v = basis_all{v1,v2,v3};
                        vindex = INDEX{v1,v2,v3};
                        nbasis_v = size(basis_v,1);
                        %
                        S = zeros(nbasis_u,nbasis_v);
                        uDZave = cell(nbasis_u,1);
                        uVLjmp = cell(nbasis_u,1);
                        for a=1:nbasis_u
                            if(uch==0)
                                uDZave{a} = squeeze((basis_u{a,DZ}(:,:,end) + 0)/2);
                                uVLjmp{a} = squeeze((basis_u{a,VL}(:,:,end)*1 + 0));
                            else
                                uDZave{a} = squeeze((basis_u{a,DZ}(:,:,1) + 0)/2);
                                uVLjmp{a} = squeeze((basis_u{a,VL}(:,:,1)*-1 + 0));
                            end
                        end
                        vDZave = cell(nbasis_v,1);
                        vVLjmp = cell(nbasis_v,1);
                        for b=1:nbasis_v   
                            if(vch==0)
                                vDZave{b} = squeeze((basis_v{b,DZ}(:,:,end) + 0)/2);
                                vVLjmp{b} = squeeze((basis_v{b,VL}(:,:,end)*1 + 0));
                            else
                                vDZave{b} = squeeze((basis_v{b,DZ}(:,:,1) + 0)/2);
                                vVLjmp{b} = squeeze((basis_v{b,VL}(:,:,1)*-1 + 0));
                            end
                        end
                        for a=1:nbasis_u
                            for b=1:nbasis_v
                                tmp1 = beta*-uDZave{a}.*vVLjmp{b}.*ww;            tmp1 = sum(tmp1(:));
                                tmp2 = beta*-vDZave{b}.*uVLjmp{a}.*ww;            tmp2 = sum(tmp2(:));
                                tmp3 = alpha/h3*(uVLjmp{a}.*vVLjmp{b}.*ww);            tmp3 = sum(tmp3(:));
                                S(a,b) = tmp1 + tmp2 + tmp3;
                            end
                        end
                        %full
                        A(uindex,vindex) = A(uindex,vindex) + S;
                        %[I,J] = ndgrid(uindex,vindex); nbtmp = numel(I); Iall(cnt+[1:nbtmp]) = I(:); Jall(cnt+[1:nbtmp]) = J(:); Sall(cnt+[1:nbtmp]) = S(:); cnt = cnt+nbtmp;
                    end
                end
                clear uDZave uVLjmp vDZave vVLjmp;
            end
        end
    end
    %Iall = Iall(1:cnt);    Jall = Jall(1:cnt);    Sall = Sall(1:cnt);    AS = sparse(Iall,Jall,Sall);
end

%sym
A = (A+A')/2; M = (M+M')/2;
%AS = (AS+AS')/2; MS = (MS+MS')/2; M = full(MS); A = full(AS);    
