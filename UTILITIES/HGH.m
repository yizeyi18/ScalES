% Generation of the HGH pseudopotential
% Currently implemented for Na, Si, C
% 
% LLIN: 
%
% 8/31/2011: Add C 
%
% 5/1/2012:  Add support for l=0,1,2. Add Bi 

res = cell(3,2);

Znucs = [6 11 14 83];
for g=1:numel(Znucs)
  Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);

  % Initialize.  
  % LLIN: IMPORTANT: Everthing is zero but r variables initialized to 1
  % to avoid overflow /underflow 
  Zion = 0;
  mass = 0;
  rloc = 1;
  C1   = 0;
  C2   = 0;
  C3   = 0;
  C4   = 0;
  r0   = 1;
  h011 = 0;
  h022 = 0;
  h033 = 0;
  r1   = 1;
  h111 = 0;
  h122 = 0;
  h133 = 0;
  r2   = 1;
  h211 = 0;
  h222 = 0;
  h233 = 0;

  % C
  if(Znuc==6)
    Zion = 4;
    mass = 12.011;
    rloc = 0.348830;
    C1 = -8.513771;
    C2 = 1.228432;
    r0 = 0.304553;
    h011 = 9.522842;
    r1 = 0.232677;
    rhocut = 2.5;
    wavcut = 2.5;
  end
  
  % Na (No semicore)
  if(Znuc==11)
    Zion = 1;
    mass = 22.989768;
    rloc = 0.885509;
    C1 = -1.238867;
    r0 = 0.661104;
    h011 = 1.847271;
    h022 = 0.582004;
    r1 = 0.857119;
    h111 = 0.471133;
    rhocut = 5.5;
    wavcut = 5.5;
  end

  %Si
  if(Znuc==14)
    Zion = 4;
    mass = 28.0855;
    rloc = 0.440000;
    C1 = -7.336103;
    r0 = 0.422738;
    h011 = 5.906928;
    h022 = 3.258196;
    r1 = 0.484278;
    h111 = 2.727013;
    rhocut = 3;
    wavcut = 3;
  end

  % Bi
  if(Znuc==83)
    Zion    = 5;
    mass    = 208.98040;
    rloc    = 0.605000;
    C1      = 6.679437;
    r0      = 0.678858;
    h011    = 1.377634;
    h022    = -0.513697;
    h033    = -0.471028;
    r1      = 0.798673;
    h111    = 0.655578;
    h122    = -0.402932;
    r2      = 0.934683;
    h211    = 0.378476;
    rhocut = 4.5;
    wavcut = 4.5;
  end

  % Derived quantities
  h012 = -1/2*sqrt(3/5)*h022;
  h013 = 1/2 * sqrt(5/21) * h033;
  h023 = -1/2 * sqrt(100/63) * h033;
  h112 = -1/2 * sqrt(5/7) * h122;
  h113 = 1/6 * sqrt(35/11) * h133;
  h123 = -1/6 * 14 / sqrt(11) * h133;
  h212 = -1/2 * sqrt(7/9) * h222;
  h213 = 1/2  * sqrt(63/143) * h233;
  h223 = -1/2 * 18 / sqrt(143) * h233;
  

  % Functional form for pseudopotentials
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

  l=0; i=3; rl=r0;
  p03fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

  l=1; i=1; rl=r1;
  p11fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

  l=1; i=2; rl=r1;
  p12fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

  l=1; i=3; rl=r1;
  p13fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
 
  l=2; i=1; rl=r2;
  p21fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

  l=2; i=2; rl=r2;
  p22fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

  l=2; i=3; rl=r2;
  p23fn = @(r) ...
    (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
    (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));


  % Reorthogonalize the h coefficient matrices
  % l=0 channel
  tmp = [h011 h012 h013;
         h012 h022 h023;
	 h013 h023 h033];
  [V0,D0] = eig(tmp);
  D0 = diag(D0);  [D0, idx] = sort(D0,'descend'); V0 = V0(:,idx);

  p0afn = @(r) (p01fn(r)*V0(1,1) + p02fn(r)*V0(2,1) + p03fn(r)*V0(3,1));
  h0a = D0(1);

  p0bfn = @(r) (p01fn(r)*V0(1,2) + p02fn(r)*V0(2,2) + p03fn(r)*V0(3,2));
  h0b = D0(2);

  p0cfn = @(r) (p01fn(r)*V0(1,3) + p02fn(r)*V0(2,3) + p03fn(r)*V0(3,3));
  h0c = D0(3);

  % l=1 channel
  tmp = [h111 h112 h113;
         h112 h122 h123;
	 h113 h123 h133];
  [V1,D1] = eig(tmp);
  D1 = diag(D1);  [D1, idx] = sort(D1,'descend'); V1 = V1(:,idx);

  p1afn = @(r) (p11fn(r)*V1(1,1) + p12fn(r)*V1(2,1) + p13fn(r)*V1(3,1));
  h1a   = D1(1);

  p1bfn = @(r) (p11fn(r)*V1(1,2) + p12fn(r)*V1(2,2) + p13fn(r)*V1(3,2));
  h1b   = D1(2); 

  p1cfn = @(r) (p11fn(r)*V1(1,3) + p12fn(r)*V1(2,3) + p13fn(r)*V1(3,3));
  h1c   = D1(3);

  % l=2 channel. 
  tmp = [h211 h212 h213;
         h212 h222 h223;
	 h213 h223 h233];
  
  [V2,D2] = eig(tmp);
  D2 = diag(D2);  [D2, idx] = sort(D2,'descend'); V2 = V2(:,idx);
  
  p2afn = @(r) (p21fn(r)*V2(1,1) + p22fn(r)*V2(2,1) + p23fn(r)*V2(3,1));
  h2a   = D2(1);

  p2bfn = @(r) (p21fn(r)*V2(1,2) + p22fn(r)*V2(2,2) + p23fn(r)*V2(3,2));
  h2b   = D2(2);

  p2cfn = @(r) (p21fn(r)*V2(1,3) + p22fn(r)*V2(2,3) + p23fn(r)*V2(3,3));
  h2c   = D2(3); 

  %sample
  stp = 0.002;
  r = [-10+stp/2:stp:10];

  Vloc = Vlocfn(r);
  p0a = p0afn(r);
  p0b = p0bfn(r);
  p0c = p0cfn(r);
  p1a = p1afn(r);
  p1b = p1bfn(r);
  p1c = p1cfn(r);
  p2a = p2afn(r);
  p2b = p2bfn(r);
  p2c = p2cfn(r);

  p0a_pp = csape(r,p0a);
  p0b_pp = csape(r,p0b);
  p0c_pp = csape(r,p0c);
  p1a_pp = csape(r,p1a);
  p1b_pp = csape(r,p1b);
  p1c_pp = csape(r,p1c);
  p2a_pp = csape(r,p2a);
  p2b_pp = csape(r,p2b);
  p2c_pp = csape(r,p2c);

  rho = rho0fn(r); 
  drho = drho0fn(r); 

  p0a_dpp = fnder(p0a_pp,1);
  dp0a = fnval(p0a_dpp,r);

  p0b_dpp = fnder(p0b_pp,1);
  dp0b = fnval(p0b_dpp,r);

  p0c_dpp = fnder(p0c_pp,1);
  dp0c = fnval(p0c_dpp,r);

  p1a_dpp = fnder(p1a_pp,1);
  dp1a = fnval(p1a_dpp,r);

  p1b_dpp = fnder(p1b_pp,1);
  dp1b = fnval(p1b_dpp,r);
 
  p1c_dpp = fnder(p1c_pp,1);
  dp1c = fnval(p1c_dpp,r);

  p2a_dpp = fnder(p2a_pp,1);
  dp2a = fnval(p2a_dpp,r);

  p2b_dpp = fnder(p2b_pp,1);
  dp2b = fnval(p2b_dpp,r);
  
  p2c_dpp = fnder(p2c_pp,1);
  dp2c = fnval(p2c_dpp,r);

  gd = find(r>0);
  Es = 1/2* sum(4*pi*r(gd).^2 .* Vloc(gd) .* rho(gd) * stp);

  cur = [Znuc mass Zion Es];

  numTerms = 21;
  spl = zeros(numel(r), numTerms);
  wgt = zeros(1,numTerms);
  typ = zeros(1,numTerms);
  cut = zeros(1,numTerms);

  cnt = 1;


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

  spl(:,cnt) = p0c(:);
  wgt(cnt) = h0c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp0c(:);
  wgt(cnt) = h0c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;

  spl(:,cnt) = p1a(:);
  wgt(cnt) = h1a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp1a(:);
  wgt(cnt) = h1a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;

  spl(:,cnt) = p1b(:);
  wgt(cnt) = h1b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp1b(:);
  wgt(cnt) = h1b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;

  spl(:,cnt) = p1c(:);
  wgt(cnt) = h1c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp1c(:);
  wgt(cnt) = h1c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;

  spl(:,cnt) = p2a(:);
  wgt(cnt) = h2a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp2a(:);
  wgt(cnt) = h2a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  
  spl(:,cnt) = p2b(:);
  wgt(cnt) = h2b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp2b(:);
  wgt(cnt) = h2b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;

  spl(:,cnt) = p2c(:);
  wgt(cnt) = h2c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
  spl(:,cnt) = dp2c(:);
  wgt(cnt) = h2c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;



  if(1)
    fprintf('Check for normalization condition for element %3d\n', Znuc);
    % Local pseudopotential
    pk = find(r>0 & r<rhocut); 
    fprintf('int rho         = %15.5e\n', sum(4*pi*r(pk).^2.*(-rho(pk))*stp) );

    % Nonlocal pseudopotential
    pk = find(r>0 & r<wavcut);
    if( abs(h0a) > 1e-10 )
      fprintf('Abs: int p0a^2  = %15.5e\n', sum(r(pk).^2.*p0a(pk).^2*stp) );
      fprintf('Rel: int p0a^2  = %15.5e\n', ...
	sum(r(pk).^2.*p0a(pk).^2*stp) / (sum(r.^2.*p0a.^2*stp)/2) );
    end
    if( abs(h0b) > 1e-10 )
      fprintf('Abs: int p0b^2  = %15.5e\n', sum(r(pk).^2.*p0b(pk).^2*stp) );
      fprintf('Rel: int p0b^2  = %15.5e\n', ...
	sum(r(pk).^2.*p0b(pk).^2*stp) / (sum(r.^2.*p0b.^2*stp)/2) );
    end
    if( abs(h0c) > 1e-10 )
      fprintf('Abs: int p0c^2  = %15.5e\n', sum(r(pk).^2.*p0c(pk).^2*stp) );
      fprintf('Rel: int p0c^2  = %15.5e\n', ...
	sum(r(pk).^2.*p0c(pk).^2*stp) / (sum(r.^2.*p0c.^2*stp)/2) );
    end
    if( abs(h1a) > 1e-10 )
      fprintf('Abs: int p1a^2  = %15.5e\n', sum(r(pk).^2.*p1a(pk).^2*stp) );
      fprintf('Rel: int p1a^2  = %15.5e\n', ...
	sum(r(pk).^2.*p1a(pk).^2*stp) / (sum(r.^2.*p1a.^2*stp)/2) );
    end
    if( abs(h1b) > 1e-10 )
      fprintf('Abs: int p1b^2  = %15.5e\n', sum(r(pk).^2.*p1b(pk).^2*stp) );
      fprintf('Rel: int p1b^2  = %15.5e\n', ...
	sum(r(pk).^2.*p1b(pk).^2*stp) / (sum(r.^2.*p1b.^2*stp)/2) );
    end
    if( abs(h1c) > 1e-10 )
      fprintf('Abs: int p1c^2  = %15.5e\n', sum(r(pk).^2.*p1c(pk).^2*stp) );
      fprintf('Rel: int p1c^2  = %15.5e\n', ...
	sum(r(pk).^2.*p1c(pk).^2*stp) / (sum(r.^2.*p1c.^2*stp)/2) );
    end
    if( abs(h2a) > 1e-10 )
      fprintf('Abs: int p2a^2  = %15.5e\n', sum(r(pk).^2.*p2a(pk).^2*stp) );
      fprintf('Rel: int p2a^2  = %15.5e\n', ...
	sum(r(pk).^2.*p2a(pk).^2*stp) / (sum(r.^2.*p2a.^2*stp)/2) );
    end
    if( abs(h2b) > 1e-10 )
      fprintf('Abs: int p2b^2  = %15.5e\n', sum(r(pk).^2.*p2b(pk).^2*stp) );
      fprintf('Rel: int p2b^2  = %15.5e\n', ...
	sum(r(pk).^2.*p2b(pk).^2*stp) / (sum(r.^2.*p2b.^2*stp)/2) );
    end
    if( abs(h2c) > 1e-10 )
      fprintf('Abs: int p2c^2  = %15.5e\n', sum(r(pk).^2.*p2c(pk).^2*stp) );
      fprintf('Rel: int p2c^2  = %15.5e\n', ...
	sum(r(pk).^2.*p2c(pk).^2*stp) / (sum(r.^2.*p2c.^2*stp)/2) );
    end

    fprintf('\n');
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
