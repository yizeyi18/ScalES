if(1)
  N1 = 2;  N2 = 1;  N3 = 1;
  h1 = 20;  h2 = 10;  h3 = 10;
  Nlbl1 = 32;  Nlbl2 = 16;  Nlbl3 = 16;
  Ndeg = 0;
  alpha = 100;
  
  if(1)
    binstr = 'singbasis';
    fid = fopen(binstr,'r');
    string = {'NumTns', {'vector',{'vector',{'DblNumTns'}}}};
    singbasis = deserialize(fid, string);
    fclose(fid);
    
    binstr = 'vtot';
    fid = fopen(binstr,'r');
    string = {'NumTns', {'DblNumTns'}};
    vtot = deserialize(fid, string);
    fclose(fid);
    
    binstr = 'pseudopot';
    fid = fopen(binstr,'r');
    string = {'vector', {'NumTns', {'DblNumTns'}}};
    pseudopot = deserialize(fid, string);
    fclose(fid);
    
    binstr = 'pseudosgn';
    fid = fopen(binstr,'r');
    string = {'vector', {'int'}};
    pseudosgn = deserialize(fid, string);
    fclose(fid);
  end
  
  nelem = N1 * N2 * N3;
  EPS = 1e-6; %SVD CUTOFF VALUE
  
  [x1,w1,P] = lglnodes(Nlbl1-1);  x1 = (x1(end:-1:1))/2;  w1 = w1(end:-1:1)/2 * h1;
  [x2,w2,P] = lglnodes(Nlbl2-1);  x2 = (x2(end:-1:1))/2;  w2 = w2(end:-1:1)/2 * h2;
  [x3,w3,P] = lglnodes(Nlbl3-1);  x3 = (x3(end:-1:1))/2;  w3 = w3(end:-1:1)/2 * h3;
  
  %%GET DATA FROM LIN'S FILE
  sing_all = cell(N1,N2,N3);
  for i1=1:N1
    for i2=1:N2
      for i3=1:N3
        old = singbasis{i1,i2,i3};
        new = cell(numel(old),4);
        for g=1:numel(old)
          new{g,1} = old{g}{1};
          new{g,2} = old{g}{2};
          new{g,3} = old{g}{3};
          new{g,4} = old{g}{4};
        end
        sing_all{i1,i2,i3} = new;
      end
    end
  end
  
  
  vtot_all = vtot;
  
  Npseudo = numel(pseudopot);
  vnl_all = pseudopot;
  
  wqsign = zeros(Npseudo,1);
  for g=1:Npseudo
    wqsign(g) = pseudosgn{g};
  end
  
  basis_all = cell(N1,N2,N3);  %Nbasis_per_cell = zeros(N1,N2,N3);
  [gg1,gg2,gg3] = ndgrid(x1*h1, x2*h2, x3*h3);
  www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]); %weight within each cell
  for i1=1:N1
    for i2=1:N2
      for i3=1:N3
        %center
        xx1 = gg1;        xx2 = gg2;        xx3 = gg3;        ee = ones(size(xx1));
        cnt = 0;
        VL=1;        DX=2;        DY=3;        DZ=4;
        %VLall = zeros(1000,Nlbl1*Nlbl2*Nlbl3);
        %DXall = zeros(1000,Nlbl1*Nlbl2*Nlbl3);
        %DYall = zeros(1000,Nlbl1*Nlbl2*Nlbl3);
        %DZalL = zeros(1000,Nlbl1*Nlbl2*Nlbl3);
        basis = cell(1000, 4);
        tmp = sing_all{i1,i2,i3};
        basis(cnt+[1:size(tmp,1)],:) = tmp;
        cnt = cnt+size(tmp,1);
        
        for d1=0:Ndeg
          for d2=0:Ndeg
            for d3=0:Ndeg
              if(d1+d2+d3<=Ndeg)
                cnt = cnt+1;
                basis{cnt,VL} = xx1.^d1 .* xx2.^d2 .* xx3.^d3;
                if(d1==0)
                  basis{cnt,DX} = 0*ee;
                else
                  basis{cnt,DX} = d1 * xx1.^(d1-1) .* xx2.^d2 .* xx3.^d3;
                end
                if(d2==0)
                  basis{cnt,DY} = 0*ee;
                else
                  basis{cnt,DY} = d2 * xx1.^d1 .* xx2.^(d2-1) .* xx3.^d3;
                end
                if(d3==0)
                  basis{cnt,DZ} = 0*ee;
                else
                  basis{cnt,DZ} = d3 * xx1.^d1 .* xx2.^d2 .* xx3.^(d3-1);
                end
              end
            end
          end
        end
        basis = basis(1:cnt,:);
        if(1)
          VLtmp = zeros(Nlbl1*Nlbl2*Nlbl3,cnt);
          DXtmp = VLtmp;
          DYtmp = VLtmp;
          DZtmp = VLtmp;
          for g=1:cnt
            VLtmp(:,g) = basis{g,1}(:);
            DXtmp(:,g) = basis{g,2}(:);
            DYtmp(:,g) = basis{g,3}(:);
            DZtmp(:,g) = basis{g,4}(:);
          end
	  if(1)
            tmp = VLtmp;
            for g=1:cnt
              tmp(:,g) = tmp(:,g) .* sqrt(www(:));
            end
	    [U,S,V] = svd(tmp,0);	    %S(1)/S(end,end)
	    gud = find(diag(S)>EPS*S(1)); 
	    disp('Effective basis ratio');
	    max(gud)/length(S)
	    G = V*inv(S); %transformation matrix
	    VLtmp = VLtmp * G;
	    DXtmp = DXtmp * G;
	    DYtmp = DYtmp * G;
	    DZtmp = DZtmp * G;
	  end
         basis = cell(cnt,4);
          for g=1:cnt
            basis{g,1} = reshape(VLtmp(:,g),[Nlbl1,Nlbl2,Nlbl3]);
            basis{g,2} = reshape(DXtmp(:,g),[Nlbl1,Nlbl2,Nlbl3]);
            basis{g,3} = reshape(DYtmp(:,g),[Nlbl1,Nlbl2,Nlbl3]);
            basis{g,4} = reshape(DZtmp(:,g),[Nlbl1,Nlbl2,Nlbl3]);
          end
        end
        %cutoff the unused space
        basis_all{i1,i2,i3} = basis;        %Nbasis_per_cell(i1,i2,i3)=size(basis,1);
      end
    end
  end
  
  %Get the INDEX set
  Ncells = N1*N2*N3;
  INDEX = cell(N1,N2,N3);
  cnt = 0;
  for i3=1:N3
    for i2=1:N2
      for i1=1:N1
        ntmp = size(basis_all{i1,i2,i3},1);
        INDEX{i1,i2,i3} = cnt+[1:ntmp];
        cnt = cnt + ntmp;
      end
    end
  end  
  Ndof = cnt;
  
  M = zeros(Ndof,Ndof);
  A = zeros(Ndof,Ndof);
  
  if(1)
    NMAX = 10*1024^2;
    Iall = zeros(NMAX,1);
    Jall = zeros(NMAX,1);
    Sall = zeros(NMAX,1);
    cnt = 0;
    %M = zeros(Ndof,Ndof);
    %the mass matrix
    www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]); %weight within each cell
    for i3=1:N3
      for i2=1:N2
        for i1=1:N1
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
          [I,J] = ndgrid(uindex,vindex);
          nbtmp = numel(I);
          Iall(cnt+[1:nbtmp]) = I(:);
          Jall(cnt+[1:nbtmp]) = J(:);
          Sall(cnt+[1:nbtmp]) = S(:);
          cnt = cnt+nbtmp;
        end
      end
    end
    Iall = Iall(1:cnt);
    Jall = Jall(1:cnt);
    Sall = Sall(1:cnt);
    MS = sparse(Iall,Jall,Sall);
  end
  
  if(1)
    NMAX = 10*1024^2;
    Iall = zeros(NMAX,1);
    Jall = zeros(NMAX,1);
    Sall = zeros(NMAX,1);
    cnt = 0;
    
    %A = zeros(Ndof,Ndof);
    %term 1
    %iterate through cells
    www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]);
    for i3=1:N3
      for i2=1:N2
        for i1=1:N1
          %only self-self interaction
          basis = basis_all{i1,i2,i3};
          nbasis = size(basis,1);
          VL=1;        DX=2;        DY=3;        DZ=4;
          S = zeros(nbasis,nbasis);
          for a=1:nbasis
            for b=1:nbasis
              tmp = 0.5*(basis{a,DX}.*basis{b,DX}+basis{a,DY}.*basis{b,DY}+basis{a,DZ}.*basis{b,DZ}).*www;
              S(a,b) = sum(tmp(:));
            end
          end
          uindex = INDEX{i1,i2,i3};
          vindex = INDEX{i1,i2,i3};
          %full
          A(uindex,vindex) = A(uindex,vindex) + S;
          [I,J] = ndgrid(uindex,vindex);
          nbtmp = numel(I);
          Iall(cnt+[1:nbtmp]) = I(:);
          Jall(cnt+[1:nbtmp]) = J(:);
          Sall(cnt+[1:nbtmp]) = S(:);
          cnt = cnt+nbtmp;
        end
      end
    end
    %term 234
    %yz faces
    ww = w2*w3'; %weight within each cell
    for i3=1:N3
      for i2=1:N2
        for i1=1:N1
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
              ubasis = basis_all{u1,u2,u3};
              uindex = INDEX{u1,u2,u3};
              nubasis = size(ubasis,1);
              if(vch==0)
                v1=p1;              v2=i2;              v3=i3;
              else
                v1=i1;              v2=i2;              v3=i3;
              end
              vbasis = basis_all{v1,v2,v3};
              vindex = INDEX{v1,v2,v3};
              nvbasis = size(vbasis,1);
              %
              S = zeros(nubasis,nvbasis);
              uDXave = cell(nubasis,1);
              uVLjmp = cell(nubasis,1);
              for a=1:nubasis
                if(uch==0)
                  uDXave{a} = squeeze((ubasis{a,DX}(end,:,:) + 0)/2);
                  uVLjmp{a} = squeeze((ubasis{a,VL}(end,:,:)*1 + 0));
                else
                  uDXave{a} = squeeze((ubasis{a,DX}(1,:,:) + 0)/2);
                  uVLjmp{a} = squeeze((ubasis{a,VL}(1,:,:)*-1 + 0));
                end
              end
              vDXave = cell(nvbasis,1);
              vVLjmp = cell(nvbasis,1);
              for b=1:nvbasis
                if(vch==0)
                  vDXave{b} = squeeze((vbasis{b,DX}(end,:,:) + 0)/2);
                  vVLjmp{b} = squeeze((vbasis{b,VL}(end,:,:)*1 + 0));
                else
                  vDXave{b} = squeeze((vbasis{b,DX}(1,:,:) + 0)/2);
                  vVLjmp{b} = squeeze((vbasis{b,VL}(1,:,:)*-1 + 0));
                end
              end
              for a=1:nubasis
                for b=1:nvbasis
                  tmp1 = .5*-uDXave{a}.*vVLjmp{b}.*ww;            tmp1 = sum(tmp1(:));
                  tmp2 = .5*-vDXave{b}.*uVLjmp{a}.*ww;            tmp2 = sum(tmp2(:));
                  tmp3 = alpha/h1*(uVLjmp{a}.*vVLjmp{b}.*ww);            tmp3 = sum(tmp3(:));
                  S(a,b) = tmp1 + tmp2 + tmp3;
                end
              end
              %full
              A(uindex,vindex) = A(uindex,vindex) + S;
              [I,J] = ndgrid(uindex,vindex);
              nbtmp = numel(I);
              Iall(cnt+[1:nbtmp]) = I(:);
              Jall(cnt+[1:nbtmp]) = J(:);
              Sall(cnt+[1:nbtmp]) = S(:);
              cnt = cnt+nbtmp;
            end
          end
          clear uDXave uVLjmp vDXave vVLjmp;
        end
      end
    end
    %zx faces
    ww = w1*w3'; %weight within each cell
    for i3=1:N3
      for i2=1:N2
        for i1=1:N1
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
              ubasis = basis_all{u1,u2,u3};
              uindex = INDEX{u1,u2,u3};
              nubasis = size(ubasis,1);
              if(vch==0)
                v1=i1;              v2=p2;              v3=i3;
              else
                v1=i1;              v2=i2;              v3=i3;
              end
              vbasis = basis_all{v1,v2,v3};
              vindex = INDEX{v1,v2,v3};
              nvbasis = size(vbasis,1);
              %
              S = zeros(nubasis,nvbasis);
              uDYave = cell(nubasis,1);
              uVLjmp = cell(nubasis,1);
              for a=1:nubasis
                if(uch==0)
                  uDYave{a} = squeeze((ubasis{a,DY}(:,end,:) + 0)/2);
                  uVLjmp{a} = squeeze((ubasis{a,VL}(:,end,:)*1 + 0));
                else
                  uDYave{a} = squeeze((ubasis{a,DY}(:,1,:) + 0)/2);
                  uVLjmp{a} = squeeze((ubasis{a,VL}(:,1,:)*-1 + 0));
                end
              end
              vDYave = cell(nvbasis,1);
              vVLjmp = cell(nvbasis,1);
              for b=1:nvbasis
                if(vch==0)
                  vDYave{b} = squeeze((vbasis{b,DY}(:,end,:) + 0)/2);
                  vVLjmp{b} = squeeze((vbasis{b,VL}(:,end,:)*1 + 0));
                else
                  vDYave{b} = squeeze((vbasis{b,DY}(:,1,:) + 0)/2);
                  vVLjmp{b} = squeeze((vbasis{b,VL}(:,1,:)*-1 + 0));
                end
              end
              for a=1:nubasis
                for b=1:nvbasis
                  tmp1 = .5*-uDYave{a}.*vVLjmp{b}.*ww;            tmp1 = sum(tmp1(:));
                  tmp2 = .5*-vDYave{b}.*uVLjmp{a}.*ww;            tmp2 = sum(tmp2(:));
                  tmp3 = alpha/h2*(uVLjmp{a}.*vVLjmp{b}.*ww);            tmp3 = sum(tmp3(:));
                  S(a,b) = tmp1 + tmp2 + tmp3;
                end
              end
              %full
              A(uindex,vindex) = A(uindex,vindex) + S;
              [I,J] = ndgrid(uindex,vindex);
              nbtmp = numel(I);
              Iall(cnt+[1:nbtmp]) = I(:);
              Jall(cnt+[1:nbtmp]) = J(:);
              Sall(cnt+[1:nbtmp]) = S(:);
              cnt = cnt+nbtmp;
            end
          end
          clear uDYave uVLjmp vDYave vVLjmp;
        end
      end
    end
    %xy faces
    ww = w1*w2'; %weight within each cell
    for i3=1:N3
      for i2=1:N2
        for i1=1:N1
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
              ubasis = basis_all{u1,u2,u3};
              uindex = INDEX{u1,u2,u3};
              nubasis = size(ubasis,1);
              if(vch==0)
                v1=i1;              v2=i2;              v3=p3;
              else
                v1=i1;              v2=i2;              v3=i3;
              end
              vbasis = basis_all{v1,v2,v3};
              vindex = INDEX{v1,v2,v3};
              nvbasis = size(vbasis,1);
              %
              S = zeros(nubasis,nvbasis);
              uDZave = cell(nubasis,1);
              uVLjmp = cell(nubasis,1);
              for a=1:nubasis
                if(uch==0)
                  uDZave{a} = squeeze((ubasis{a,DZ}(:,:,end) + 0)/2);
                  uVLjmp{a} = squeeze((ubasis{a,VL}(:,:,end)*1 + 0));
                else
                  uDZave{a} = squeeze((ubasis{a,DZ}(:,:,1) + 0)/2);
                  uVLjmp{a} = squeeze((ubasis{a,VL}(:,:,1)*-1 + 0));
                end
              end
              vDZave = cell(nvbasis,1);
              vVLjmp = cell(nvbasis,1);
              for b=1:nvbasis   
                if(vch==0)
                  vDZave{b} = squeeze((vbasis{b,DZ}(:,:,end) + 0)/2);
                  vVLjmp{b} = squeeze((vbasis{b,VL}(:,:,end)*1 + 0));
                else
                  vDZave{b} = squeeze((vbasis{b,DZ}(:,:,1) + 0)/2);
                  vVLjmp{b} = squeeze((vbasis{b,VL}(:,:,1)*-1 + 0));
                end
              end
              for a=1:nubasis
                for b=1:nvbasis
                  tmp1 = .5*-uDZave{a}.*vVLjmp{b}.*ww;            tmp1 = sum(tmp1(:));
                  tmp2 = .5*-vDZave{b}.*uVLjmp{a}.*ww;            tmp2 = sum(tmp2(:));
                  tmp3 = alpha/h3*(uVLjmp{a}.*vVLjmp{b}.*ww);            tmp3 = sum(tmp3(:));
                  S(a,b) = tmp1 + tmp2 + tmp3;
                end
              end
              %full
              A(uindex,vindex) = A(uindex,vindex) + S;
              [I,J] = ndgrid(uindex,vindex);
              nbtmp = numel(I);
              Iall(cnt+[1:nbtmp]) = I(:);
              Jall(cnt+[1:nbtmp]) = J(:);
              Sall(cnt+[1:nbtmp]) = S(:);
              cnt = cnt+nbtmp;
          end
          end
          clear uDZave uVLjmp vDZave vVLjmp;
        end
      end
    end
    %term 5, potential
    % www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]); %weight within each cell
    for i3=1:N3
      for i2=1:N2
        for i1=1:N1
          basis = basis_all{i1,i2,i3};
          nbasis = size(basis,1);
          VL=1;        DX=2;        DY=3;        DZ=4;
          vtot = vtot_all{i1,i2,i3};
          S = zeros(nbasis,nbasis);
          for a=1:nbasis
            for b=1:nbasis
              tmp = (basis{a,VL}.*basis{b,VL}).*vtot.*www;
              S(a,b) = sum(tmp(:));
            end
          end
          uindex = INDEX{i1,i2,i3};
          vindex = INDEX{i1,i2,i3};
          %
          A(uindex,vindex) = A(uindex,vindex) + S;
          [I,J] = ndgrid(uindex,vindex);
          nbtmp = numel(I);
          Iall(cnt+[1:nbtmp]) = I(:);
          Jall(cnt+[1:nbtmp]) = J(:);
          Sall(cnt+[1:nbtmp]) = S(:);
          cnt = cnt+nbtmp;
        end
      end
    end
    
    %term 6, pseudopotential
    % www = reshape(reshape(w1*w2',[Nlbl1*Nlbl2,1])*w3', [Nlbl1 Nlbl2 Nlbl3]); %weight within each cell
    for g=1:Npseudo
      vnl_cur = vnl_all{g};
      sign = wqsign(g);
      %
      inner = cell(N1,N2,N3);
      for i3=1:N3
        for i2=1:N2
          for i1=1:N1
            if(isempty(vnl_cur{i1,i2,i3})==0)
              basis = basis_all{i1,i2,i3};
              index = INDEX{i1,i2,i3};
              nbasis = size(basis,1);
              mid = zeros(nbasis,1);
              for a=1:nbasis
                tmp = (basis{a,VL}.*vnl_cur{i1,i2,i3}).*www;
                mid(a) = sum(tmp(:));
              end
              inner{i1,i2,i3} = mid;
            end
          end
        end
      end
      for i3=1:N3
        for i2=1:N2
          for i1=1:N1
            if(isempty(vnl_cur{i1,i2,i3})==0)
              uindex = INDEX{i1,i2,i3};
              nubasis = size(basis_all{i1,i2,i3},1);
              for j3=1:N3
                for j2=1:N2
                  for j1=1:N1
                    if(isempty(vnl_cur{j1,j2,j3})==0)
                      vindex = INDEX{j1,j2,j3};
                      nvbasis = size(basis_all{j1,j2,j3},1);
                      S = zeros(nubasis,nvbasis);
                      for a=1:nubasis
                        for b=1:nvbasis
                          %tmp1 = (ubasis{a,VL}.*vnl_cur{i1,i2,i3}).*www;                        tmp1 = sum(sum(sum(tmp1)));
                          tmp1 = inner{i1,i2,i3}(a);
                          tmp2 = inner{j1,j2,j3}(b);
                          S(a,b) = sign * (tmp1*tmp2);
                        end
                      end
                      A(uindex,vindex) = A(uindex,vindex) + S;
                      [I,J] = ndgrid(uindex,vindex);
                      nbtmp = numel(I);
                      Iall(cnt+[1:nbtmp]) = I(:);
                      Jall(cnt+[1:nbtmp]) = J(:);
                      Sall(cnt+[1:nbtmp]) = S(:);
                      cnt = cnt+nbtmp;
                    end
                  end
                end
              end
            end
          end
        end
      end
    end  % for g
  
    %tmp = M-M';  norm(tmp(:))
    %tmp = A-A';  norm(tmp(:))
    Iall = Iall(1:cnt);
    Jall = Jall(1:cnt);
    Sall = Sall(1:cnt);
    AS = sparse(Iall,Jall,Sall);
    fprintf('done assemble\n');
  end
  
  %eig
  if(1)
    Neigs = 30;
    
    AS = (AS + AS')/2;
    MS = (MS + MS')/2;
    
    es = eig(full(AS),full(MS));
    es = sort(real(es));
    es(1:Neigs)

    
    tic; [vs,es] = eigs(AS,MS,Neigs,'SA'); toc;
    [es, ind] =  sort(real(diag(es)));
    es
    vs = vs(:,ind);
    condest(AS)
    condest(MS)
  end

  %get eigenfuncs
  if(0)
    denfun = cell(N1,N2,N3);
    for i1=1:N1
      for i2=1:N2
        for i3=1:N3
          denfun{i1,i2,i3} = zeros(Nlbl1,Nlbl2,Nlbl3);
        end
      end
    end
    eigfuns = cell(Neigs,1);
    for g=1:Neigs
      v = vs(:,g);
      cur = cell(N1,N2,N3);
      for i1=1:N1
        for i2=1:N2
          for i3=1:N3
            basis = basis_all{i1,i2,i3};
            index = INDEX{i1,i2,i3};
            nbasis = size(basis,1);
            vcell = v(index);
            %
            tmp = zeros(Nlbl1,Nlbl2,Nlbl3);
            for a=1:nbasis
              tmp = tmp + basis{a,VL}*vcell(a); %linear combination
            end
            cur{i1,i2,i3} = tmp;
            denfun{i1,i2,i3} = denfun{i1,i2,i3} + (abs(tmp)).^2;
          end
        end
      end
      eigfuns{g} = cur;
    end
  end
  
  %% interp
  if(0)
    gh1 = N1*h1; gh2 = N2*h2; gh3 = N3*h3;
    
    gx1 = linspace(0, gh1, gn1+1); gx1 = gx1(1:end-1)';
    gx2 = linspace(0, gh2, gn2+1); gx2 = gx2(1:end-1)';
    gx3 = linspace(0, gh3, gn3+1); gx3 = gx3(1:end-1)';
    
    eigfunsuni = cell(Neigs,1);
    
    for g = 1:Neigs
      eigfunsuni{g} = zeros(gn1, gn2, gn3);
    end
    
    for i1 = 1:N1
      for i2 = 1:N2
        for i3 = 1:N3
          
          xx1 = (i1-1/2)*h1 + x1 * h1;
          xx2 = (i2-1/2)*h2 + x2 * h2;
          xx3 = (i3-1/2)*h3 + x3 * h3;
        
          Transx = lag1dm(gx1, xx1, Nlbl1-1);
          Transy = lag1dm(gx2, xx2, Nlbl2-1);
          Transz = lag1dm(gx3, xx3, Nlbl3-1);
          
          for g = 1:Neigs
            tmp = eigfuns{g}{i1,i2,i3};
            tmp = reshape(tmp, Nlbl1, Nlbl2*Nlbl3);
            tmp = Transx * tmp;
            tmp = reshape(tmp, gn1, Nlbl2, Nlbl3);
            tmp = permute(tmp, [2 3 1]);
            tmp = reshape(tmp, Nlbl2, Nlbl3*gn1);
            tmp = Transy * tmp;
            tmp = reshape(tmp, gn2, Nlbl3, gn1);
            tmp = permute(tmp, [2 3 1]);
            tmp = reshape(tmp, Nlbl3, gn1*gn2);
            tmp = Transz * tmp;
            tmp = reshape(tmp, gn3, gn1, gn2);
            tmp = permute(tmp, [2 3 1]);
            eigfunsuni{g} = eigfunsuni{g} + tmp;
          end
        end
      end 
    end 
  end

end


