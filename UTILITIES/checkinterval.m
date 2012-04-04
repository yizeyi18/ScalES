function [xyzmap, isin] = checkinterval(xyz, posstart, posend, L)
isin = 0;
xyzmap = xyz;
if( xyzmap >= posstart & xyzmap <= posend )
  isin = 1;
  xyzmap = xyzmap-posstart;
  return;
end
xyzmap = xyz + L;
if( xyzmap >= posstart & xyzmap <= posend )
  isin = 1;
  xyzmap = xyzmap-posstart;
  return;
end
xyzmap = xyz - L;
if( xyzmap >= posstart & xyzmap <= posend )
  isin = 1;
  xyzmap = xyzmap-posstart;
  return;
end

