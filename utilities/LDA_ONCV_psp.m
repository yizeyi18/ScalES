% Generate pseudopotential from psp8 file format to the binary format
% that can be read by PWDFT/DGDFT (2016 or later).

Znucs = [1 3];
res = cell(length(Znucs),2);

for g=1:length(Znucs)
  Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);

  switch(Znuc)
    case 1 % H
      mass = 1.00794;
      rhocut  = 2.0;
      wavcut  = 1.5;
      ppFile = './ONCV_Pask/20161112/H/H.LDA.psp8';
      pprholocFile = './ONCV_Pask/20161112/H/H.LDA.rholoc';
    case 3 % Li
      mass = 6.941;
      rhocut  = 3.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20161112/Li/Li.LDA.psp8';
      pprholocFile = './ONCV_Pask/20161112/Li/Li.LDA.rholoc';
  end

  pp = psp8read( [], ppFile );
  if( Znuc ~= pp.anum )
    error('Znuc != pp.anum, reading the wrong pseudopotential file');
  end
  Zion = pp.venum;

  rholocData = load(pprholocFile);
  % Note rho is positive
  sprholoc = csape(rholocData(:,1), rholocData(:,2));

  r = pp.r;
  rho = fnval(sprholoc, r);
  Vloc = pp.vloc;

  gd = find(r>0);
  % Note the minus sign
  Es = 1/2* sum(4*pi*r(gd).^2 .* Vloc(gd) .* (-rho(gd)) .* pp.rab(gd));

  cur = [Znuc mass Zion Es];

  spl = zeros(numel(r), 0);
  wgt = zeros(1,0);
  typ = zeros(1,0);
  cut = zeros(1,0);

  cnt = 1;

  spl(:,cnt) = r(:);
  wgt(cnt) = -1;    typ(cnt) =  9;    cut(cnt) = rhocut;    cnt=cnt+1;

  % local pseudopotential, derivative padded by zero and not used
  spl(:,cnt) = rho(:);
  wgt(cnt) = -1;   typ(cnt)  =  99;    cut(cnt) = rhocut;    cnt=cnt+1;
  spl(:,cnt) = zeros(size(rho));
  wgt(cnt) = -1;   typ(cnt)  =  99;  cut(cnt) = rhocut; cnt=cnt+1;

  % nonlocal pseudopotential, derivative padded by zero and not used
  for l = 1 : pp.nonloc.nbeta
    % Scale pp.nonloc.beta(:,l) by 1./r to be compatible with DGDFT
    gd = find(pp.r>0);
    pp.nonloc.beta(gd,l) = pp.nonloc.beta(gd,l)./(pp.r(gd));
    pp.nonloc.beta(1,l) =  pp.nonloc.beta(2,l); 
    spl(:,cnt) = pp.nonloc.beta(:,l);
    wgt(cnt)   = pp.nonloc.dij(l,l);
    typ(cnt)   = pp.nonloc.lll(l);
    cut(cnt)   = wavcut;
    cnt = cnt+1;

    spl(:,cnt) = zeros(size(pp.nonloc.beta(:,l)));
    wgt(cnt)   = pp.nonloc.dij(l,l);
    typ(cnt)   = pp.nonloc.lll(l);
    cut(cnt)   = wavcut;
    cnt = cnt+1;
  end


  if(1)
    fprintf('Check for normalization condition for element %3d\n', Znuc);
    fprintf('Total number of pseudopotentials terms %3d\n', cnt-1);

    % Local pseudopotential
    pk = find(r>0 & r<rhocut); 
    fprintf('int rho         = %15.10e\n', ...
      sum(4*pi*r(pk).^2.*(rho(pk)).*pp.rab(pk)));

    % Nonlocal pseudopotential
    for l = 1 : pp.nonloc.nbeta
      pk = find(r>0 & r<wavcut);
      sumpp = sum(r.^2.*(pp.nonloc.beta(:,l)).^2.*pp.rab);
      sumppcut = sum(r(pk).^2.*(pp.nonloc.beta(pk,l)).^2.*pp.rab(pk));
      fprintf('Abs: int p^2  = %15.10e\n', sumpp);
      fprintf('Rel: int p^2  = %15.10e\n', sumppcut / sumpp );
    end

    fprintf('\n');
  end

  ess = {cur spl wgt typ cut};

  res{g,1} = Znuc;
  res{g,2} = ess;

end

if(1)
  binstr = sprintf('ONCV.bin');
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
