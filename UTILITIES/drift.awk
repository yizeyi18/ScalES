BEGIN{
  N=0;
}

/Eatomkin/{
  Eatomkin = $3; getline;
  DFTEtot  = $3; getline;
  DFTEfree = $3; getline;
  if(N == 0){
    HconvEtot0 = Eatomkin + DFTEtot;
    HconvEfree0 = Eatomkin + DFTEfree;
  }
  else{
    HconvEtot   = Eatomkin + DFTEtot;
    HconvEfree  = Eatomkin + DFTEfree;
    driftEtot = (HconvEtot - HconvEtot0) / HconvEtot0;
    driftEfree = (HconvEfree - HconvEfree0) / HconvEfree0;
    print driftEtot " " driftEfree;
  }
  N++;
}
  

