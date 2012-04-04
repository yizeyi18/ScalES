%just for Na and Si
%add C 
%
%Revision: 8/31/2011 (LL)
res = cell(3,2);

Znucs = [6 11 14];
for g=1:numel(Znucs)
    Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);
    %just for C, Na and Si
    if(Znuc==6)
        Zion = 4;
        mass = 12.011;
        rloc = 0.348830;
        C1 = -8.513771;
        C2 = 1.228432;
        C3 = 0;
        C4 = 0;
        r0 = 0.304553;
        h011 = 9.522842;
        h022 = 0;
        h012 = -0.5*sqrt(3/5)*h022;
        r1 = 0.232677;
        h111 = 0;
    end
    if(Znuc==11)
        Zion = 1;
        mass = 22.989768;
        rloc = 0.885509;
        C1 = -1.238867;
        C2 = 0;
        C3 = 0;
        C4 = 0;
        r0 = 0.661104;
        h011 = 1.847271;
        h022 = 0.582004;
        h012 = -0.5*sqrt(3/5)*h022;
        r1 = 0.857119;
        h111 = 0.471133;
    end
    if(Znuc==14)
        Zion = 4;
        mass = 28.0855;
        rloc = 0.440000;
        C1 = -7.336103;
        C2 = 0;
        C3 = 0;
        C4 = 0;
        r0 = 0.422738;
        h011 = 5.906928;
        h022 = 3.258196;
        h012 = -0.5*sqrt(3/5)*h022;
        r1 = 0.484278;
        h111 = 2.727013;
    end
    
    Vlocfn = @(r) ...
           (-Zion./r.*erf(r./(sqrt(2)*rloc)) + ...
            exp(-1/2*(r./rloc).^2).*(C1+C2*(r/rloc).^2+C3*(r/ ...
                                                      rloc).^4+C4*(r./rloc).^6));

    rho0fn = @(r) ...
           (1/(4*pi).*(-sqrt(2/pi)*Zion./rloc.^3.*exp(-1/2*(r./rloc).^2) ...
                       + exp(-1/2*(r./rloc).^2).* ...
                       ((-6*C2+3*C1)/rloc.^2 + (-20*C3+7*C2-C1).*r.^2/rloc.^4 ...
                        + (-42*C4+11*C3-C2).*r.^4/rloc.^6 + ...
                        (15*C4-C3).*r.^6/rloc.^8 - C4.*r.^8./rloc.^10)));
    
    drho0fn = @(r) ...
              (1/(4*pi).*(sqrt(2/pi)*Zion*r./rloc.^5.*exp(-1/2*(r./rloc).^2) ...
                          + exp(-1/2*(r./rloc).^2).* ...
                          ((-40*C3+20*C2-5*C1)*r/rloc.^4 + ...
                           (-168*C4+64*C3-11*C2+C1).*r.^3/rloc.^6 + ...
                           (132*C4-17*C3+C2).*r.^5/rloc.^8 + ...
                           (-23*C4+C3).*r.^7/rloc.^10 + C4.*r.^9./ ...
                           rloc.^12)));
    
    
    l=0; i=1; rl=r0;
    p01fn = @(r) ...
          (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
          (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
    
    l=0; i=2; rl=r0;
    p02fn = @(r) ...
          (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
          (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
    
    l=1; i=1; rl=r1;
    p11fn = @(r) ...
          (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
          (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
    
    %reortho
    tmp = [h011 h012; h012 h022];
    [V,D] = eig(tmp);
    
    p0afn = @(r) (p01fn(r)*V(1,1) + p02fn(r)*V(2,1));
    h0a = D(1,1);
    
    p0bfn = @(r) (p01fn(r)*V(1,2) + p02fn(r)*V(2,2));
    h0b = D(2,2);
    
    %sample
    stp = 0.002;
    r = [-10+stp/2:stp:10];
    
    Vloc = Vlocfn(r);
    rVloc = r.*Vloc;
    p01 = p01fn(r);
    p02 = p02fn(r);
    p0a = p0afn(r);
    p0b = p0bfn(r);
    p11 = p11fn(r);
    
    rVloc_pp = csape(r,rVloc);
    p0a_pp = csape(r,p0a);
    p0b_pp = csape(r,p0b);
    p11_pp = csape(r,p11);
    
    rVloc_dpp = fnder(rVloc_pp,1);
    rVloc_ddpp = fnder(rVloc_pp,2);
    rVloc_dddpp = fnder(rVloc_pp,3);
    
    %tmp = - 1./(4*pi*r) .* fnval(rVloc_ddpp,r);    fprintf(1, 'rho sum %d\n', sum(4*pi*r.^2.*tmp*stp)/2);
    rho = rho0fn(r); %norm(tmp-rho)/norm(rho)
    
    %tmp = 1/2*(fnval(rVloc_dddpp,r+eps)+fnval(rVloc_dddpp,r-eps));
    %tmp = - (1./(4*pi*r).*tmp - 1./(4*pi*r.^2).*fnval(rVloc_ddpp,r));
    drho = drho0fn(r); %norm(tmp-drho)/norm(drho)
    
    p0a_dpp = fnder(p0a_pp,1);
    dp0a = fnval(p0a_dpp,r);
    
    p0b_dpp = fnder(p0b_pp,1);
    dp0b = fnval(p0b_dpp,r);
    
    p11_dpp = fnder(p11_pp,1);
    dp11 = fnval(p11_dpp,r);
    
    gd = find(r>0);
    Es = 1/2* sum(4*pi*r(gd).^2 .* Vloc(gd) .* rho(gd) * stp);
    
    cur = [Znuc mass Zion Es];
    
    spl = zeros(numel(r), 9);
    wgt = zeros(1,9);
    typ = zeros(1,9);
    cut = zeros(1,9);
    
    cnt = 1;
    
    if(Znuc==6)
      rhocut = 2.5;
      wavcut = 2.5;
    end
    if(Znuc==11)
      rhocut = 5.5;
      wavcut = 5.5;
    end
    if(Znuc==14)
      rhocut = 3;
      wavcut = 3;
    end
    
    spl(:,cnt) = r(:);
    wgt(cnt) = -1;    typ(cnt) = -1;    cut(cnt) = rhocut;    cnt=cnt+1;
    spl(:,cnt) = -rho(:);
    wgt(cnt) = -1;   typ(cnt) = -1;    cut(cnt) = rhocut;    cnt=cnt+1;
    spl(:,cnt) = -drho(:);
    wgt(cnt) = -1;   typ(cnt) = -1;  cut(cnt) = rhocut; cnt=cnt+1;
    spl(:,cnt) = p0a(:);
    wgt(cnt) = h0a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
    spl(:,cnt) = dp0a(:);
    wgt(cnt) = h0a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
    spl(:,cnt) = p0b(:);
    wgt(cnt) = h0b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
    spl(:,cnt) = dp0b(:);
    wgt(cnt) = h0b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
    spl(:,cnt) = p11(:);
    wgt(cnt) = h111;   typ(cnt) = 1; cut(cnt) = wavcut; cnt=cnt+1;
    spl(:,cnt) = dp11(:);
    wgt(cnt) = h111;   typ(cnt) = 1; cut(cnt) = wavcut; cnt=cnt+1;
    
    if(1)
        %sum(4*pi*r.^2.*(-rho)*stp)/2
        %sum(r.^2.*p0a.^2*stp)/2
        %sum(r.^2.*p0b.^2*stp)/2
        %sum(r.^2.*p11.^2*stp)/2
        pk = find(r>0 & r<rhocut);        sum(4*pi*r(pk).^2.*(-rho(pk))*stp)
        pk = find(r>0 & r<wavcut);        sum(r(pk).^2.*p11(pk).^2*stp)
    end
    
    ess = {cur spl wgt typ cut};
    
    res{g,1} = Znuc;
    res{g,2} = ess;
end

if(1)
    binstr = sprintf('HGH.bin');
    fid = fopen(binstr, 'w');
    string = {'map', ...
              {'int'}, ...
              {'tuple', ...
               {'DblNumVec'}, ...
               {'DblNumMat'}, ...
               {'DblNumVec'}, ...
               {'IntNumVec'}, ...
               {'DblNumVec'}, ...
              }...
             };
    serialize(fid,res,string);
    fclose(fid);
end

if(1)
    fid = fopen(binstr,'r');
    restst = deserialize(fid, string);
    fclose(fid);
end

