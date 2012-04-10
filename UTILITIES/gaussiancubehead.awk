BEGIN{
  print "Gaussian cube format produced by convcube";
  print "X: Outer loop Y: Middle loop Z: Inner loop";
}
/Super_Cell/{
  Ls1 = $3;
  Ls2 = $4;
  Ls3 = $5;
}
/Grid_Size/{
  Ns1 = $3;
  Ns2 = $4;
  Ns3 = $5;
}
/Total number of atoms/{
  NATOM = $6;
}

/COORDINATES/{
  hs1 = Ls1 / Ns1;
  hs2 = Ls2 / Ns2;
  hs3 = Ls3 / Ns3;
  printf("%9d %12.6f %12.6f %12.6f\n", NATOM, 0,0,0);
  printf("%9d %12.6f %12.6f %12.6f\n", Ns1, hs1, 0, 0);
  printf("%9d %12.6f %12.6f %12.6f\n", Ns2, 0, hs2, 0);
  printf("%9d %12.6f %12.6f %12.6f\n", Ns3, 0, 0, hs3);

  for(i=1; i<=NATOM; i++){
    getline;
    printf("%9d %12.6f %12.6f %12.6f %12.6f\n", $2, 0, $3, $4, $5);
  }
}
