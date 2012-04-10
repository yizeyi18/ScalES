# Split the Gaussian cube file into the positve and the negative part.
BEGIN{
  split(ARGV[1],fileNameSplit,".");
  fileNamePositive=fileNameSplit[1]"_POS.cub"
  fileNameNegative=fileNameSplit[1]"_NEG.cub"
  # Create empty files
  printf("") > fileNamePositive;
  printf("") > fileNameNegative;
}
{ 
  if( (NR < 3) || (NF > 1) ){
    print $0 >> fileNamePositive;
    print $0 >> fileNameNegative;
  }
  else{
    if( $1 >= 0 ){
      print $0 >> fileNamePositive;
      printf("%12.5f\n", 0.0) >> fileNameNegative;
    }
    else{
      printf("%12.5f\n", 0.0) >> fileNamePositive;
      print $0 >> fileNameNegative;
    }
  }
}

END{
}

