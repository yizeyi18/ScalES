
#ifndef _SCFDG_UPPER_END_SPECTRUM_HPP_
#define _SCFDG_UPPER_END_SPECTRUM_HPP_

namespace dgdft{
  
  void  LOBPCG_Hmat_top_serial(DblNumMat& Hmat,
			       DblNumMat& Xmat,
			       DblNumVec& eig_vals_Xmat,
			       Int eigMaxIter, Real eigTolerance)
  {

 
  
    // Do some basic "sanity tests" here
    if( (Hmat.m() != Hmat.n()) ||
	(Hmat.m() != Xmat.m()) ||
	(Xmat.m() <= Xmat.n())
	)
      {
	statusOFS << std::endl << " Error !! Hmat and Xmat are not dimensioned properly !";
	statusOFS << std::endl << " Hmat.m = " << Hmat.m() << " Hmat.n = " << Hmat.n();
	statusOFS << std::endl << " Xmat.m = " << Xmat.m() << " Xmat.n = " << Xmat.n();
    
	abort();
      }

    statusOFS << std::endl << std::endl;
    // *********************************************************************
    // Initialization
    // *********************************************************************
  
    Int ntot = Hmat.m();
    Int ncom = 1; // Fixed to be 1 here
    Int noccLocal = Xmat.n(); // Same values in serial version
    Int noccTotal = Xmat.n(); // Serial version

    Int height = ntot * ncom;
    Int widthTotal = noccTotal;
    Int widthLocal = noccLocal;
    Int width = noccTotal;
    Int numEig = Xmat.n();
    Int lda = 3 * width;
 


    Real timeSta, timeEnd;
    Real timeGemmT = 0.0;
    Real timeGemmN = 0.0;
    Real timeTrsm = 0.0;
    Real timeSpinor = 0.0;
    Real timeMpirank0 = 0.0;
    Int  iterGemmT = 0;
    Int  iterGemmN = 0;
    Int  iterTrsm = 0;
    Int  iterSpinor = 0;
    Int  iterMpirank0 = 0;



    // S = ( X | W | P ) is a triplet used for LOBPCG.  
    // W is the preconditioned residual
    DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
    // AMat = S' * (AS),  BMat = S' * S
    // 
    // AMat = (X'*AX   X'*AW   X'*AP)
    //      = (  *     W'*AW   W'*AP)
    //      = (  *       *     P'*AP)
    //
    // BMat = (X'*X   X'*W   X'*P)
    //      = (  *    W'*W   W'*P)
    //      = (  *      *    P'*P)
    //
    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
    DblNumMat  AMatT1( 3*width, 3*width );
    // AMatSave and BMatSave are used for restart
    // Temporary buffer array.
    // The unpreconditioned residual will also be saved in Xtemp
    DblNumMat  XTX( width, width ), Xtemp( height, widthLocal );

    // rexNorm Grobal matrix  similar to numEig 
    DblNumVec  resNormLocal ( widthLocal ); 
    SetValue( resNormLocal, 0.0 );
    DblNumVec  resNorm( width );
    SetValue( resNorm, 0.0 );
    Real       resMax, resMin;

    // For convenience
    DblNumMat  X( height, widthLocal, false, S.VecData(0) );
    DblNumMat  W( height, widthLocal, false, S.VecData(widthLocal) );
    DblNumMat  P( height, widthLocal, false, S.VecData(2*widthLocal) );
    DblNumMat AX( height, widthLocal, false, AS.VecData(0) );
    DblNumMat AW( height, widthLocal, false, AS.VecData(widthLocal) );
    DblNumMat AP( height, widthLocal, false, AS.VecData(2*widthLocal) );

    //Int info;
    bool isRestart = false;
    // numSet = 2    : Steepest descent (Davidson), only use (X | W)
    //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
    Int numSet = 2;

    // numLocked is the number of converged vectors
    Int numLockedLocal = 0, numLockedSaveLocal = 0;
    Int numLockedTotal = 0, numLockedSaveTotal = 0; 
    Int numLockedSave = 0;
    // numActive = width - numLocked
    Int numActiveLocal = 0;
    Int numActiveTotal = 0;

    // Real lockTolerance = std::min( eigTolerance_, 1e-2 );

    const Int numLocked = 0;  // Never perform locking in this version
    const Int numActive = width;

    bool isConverged = false;

    // Initialization
    SetValue( S, 0.0 );
    SetValue( AS, 0.0 );

    DblNumVec  eigValS(lda);
    SetValue( eigValS, 0.0 );
    DblNumVec  sigma2(lda);
    DblNumVec  invsigma(lda);
    SetValue( sigma2, 0.0 );
    SetValue( invsigma, 0.0 );

    // Initialize X by the data in Xmat
    lapack::Lacpy( 'A', height, width, Xmat.Data(), height, 
		   X.Data(), height );



    // *********************************************************************
    // Main loop
    // *********************************************************************

    // Orthogonalization through Cholesky factorization
    blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
		height, X.Data(), height, 0.0, XTX.Data(), width );
 

    // X'*X=U'*U
    lapack::Potrf( 'U', width, XTX.Data(), width );


    // X <- X * U^{-1} is orthogonal
    blas::Trsm( 'R', 'U', 'N', 'N', height, width, 1.0, XTX.Data(), width, 
		X.Data(), height );
 

    // Applying the negative of the Hamiltonian matrix : AX = -A * X
    {
      blas::Gemm( 'N', 'N', Hmat.m(), Xmat.n(), Hmat.n(), 
		  -1.0, 
		  Hmat.Data(), Hmat.m(),
		  Xmat.Data(), Xmat.m(), 
		  0.0, AX.Data(), AX.m() );
  
    }

 
    // Start the main loop
    Int iter;
    for( iter = 1; iter < eigMaxIter; iter++ )
      {
   
	if( iter == 1 || isRestart == true )
	  numSet = 2;
	else
	  numSet = 3;

	SetValue( AMat, 0.0 );
	SetValue( BMat, 0.0 );

	// Rayleigh Ritz in Q = span( X, gradient )

	// XTX <- X' * (AX)
	blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(),
		    height, AX.Data(), height, 0.0, XTX.Data(), width );

	lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );
  
	// Compute the residual.
	// R <- AX - X*(X'*AX)
	lapack::Lacpy( 'A', height, width, AX.Data(), height, Xtemp.Data(), height );
   
	blas::Gemm( 'N', 'N', height, width, width, -1.0, 
		    X.Data(), height, AMat.Data(), lda, 1.0, Xtemp.Data(), height );
 
	// Compute the norm of the residual
	SetValue( resNorm, 0.0 );
	for( Int k = 0; k < widthLocal; k++ )
	  {
	    resNorm( k ) = std::sqrt( Energy(DblNumVec(height, false, Xtemp.VecData(k)))); 
	  }

     
	resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
	resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

	statusOFS << " Serial LOBPCG iter = " << iter << " maxRes = " << resMax << std::endl;

	//statusOFS << "resNorm = " << resNorm << std::endl;
	//statusOFS << "maxRes  = " << resMax  << std::endl;
	//statusOFS << "minRes  = " << resMin  << std::endl;

	if( resMax < eigTolerance ){
	  isConverged = true;;
	  break;
	}

	//numActive = width - numLocked;
	numActiveLocal = widthLocal - numLockedLocal;
	numActiveTotal = widthTotal - numLockedTotal;
 
	// If the number of locked vectors goes down, perform steepest
	// descent rather than conjugate gradient
	// if( numLockedTotal < numLockedSaveTotal )
	//  numSet = 2;

	// Compute the preconditioned residual W = T*R.
	// Here, no preconditioner is applied
	{
	  lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, W.Data(), height );
	}

	//    Real normLocal; 
	Real norm; 

	norm = 0.0; 
	// Normalize the preconditioned residual
	for( Int k = numLockedLocal; k < widthLocal; k++ ){
	  norm = Energy(DblNumVec(height, false, W.VecData(k)));
	  norm = std::sqrt( norm );
	  blas::Scal( height, 1.0 / norm, W.VecData(k), 1 );
	}

	norm = 0.0; 
	// Normalize the conjugate direction
	if( numSet == 3 ){
	  for( Int k = numLockedLocal; k < widthLocal; k++ ){
	    norm = Energy(DblNumVec(height, false, P.VecData(k)));
	    norm = std::sqrt( norm );
	    blas::Scal( height, 1.0 / norm, P.VecData(k), 1 );
	    blas::Scal( height, 1.0 / norm, AP.VecData(k), 1 );
	  }
	}

	// Compute AMat

	// Applying the negative of the Hamiltonian matrix : AX = -A * X
	{
	  blas::Gemm( 'N', 'N', Hmat.m(), W.n(), Hmat.n(), 
		      -1.0, 
		      Hmat.Data(), Hmat.m(),
		      W.Data(), W.m(), 
		      0.0, AW.Data(), AW.m() );
	}

	// Compute X' * (AW)
	// Instead of saving the block at &AMat(0,width+numLocked), the data
	// is saved at &AMat(0,width) to guarantee a continuous data
	// arrangement of AMat.  The same treatment applies to the blocks
	// below in both AMat and BMat.
	blas::Gemm( 'T', 'N', width, numActive, height, 1.0, X.Data(),
		    height, AW.VecData(numLocked), height, 
		    0.0, &AMat(0,width), lda );


	// Compute W' * (AW)
	blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
		    W.VecData(numLocked), height, AW.VecData(numLocked), height, 
		    0.0, &AMat(width, width), lda );


	if( numSet == 3 ){
	  // Compute X' * (AP)
	  blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
		      X.Data(), height, AP.VecData(numLocked), height, 
		      0.0, &AMat(0, width+numActive), lda );


	  // Compute W' * (AP)
	  blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
		      W.VecData(numLocked), height, AP.VecData(numLocked), height, 
		      0.0, &AMat(width, width+numActive), lda );


	  // Compute P' * (AP)
	  blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
		      P.VecData(numLocked), height, AP.VecData(numLocked), height, 
		      0.0, &AMat(width+numActive, width+numActive), lda );

	}


 
	// Compute BMat (overlap matrix)

	// Compute X'*X
	blas::Gemm( 'T', 'N', width, width, height, 1.0, 
		    X.Data(), height, X.Data(), height, 
		    0.0, &BMat(0,0), lda );



	// Compute X'*W
	blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
		    X.Data(), height, W.VecData(numLocked), height,
		    0.0, &BMat(0,width), lda );



	// Compute W'*W
	blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
		    W.VecData(numLocked), height, W.VecData(numLocked), height,
		    0.0, &BMat(width, width), lda );




	if( numSet == 3 ){
	  // Compute X'*P
	  blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
		      X.Data(), height, P.VecData(numLocked), height, 
		      0.0, &BMat(0, width+numActive), lda );



	  // Compute W'*P
	  blas::Gemm('T', 'N', numActive, numActive, height, 1.0,
		     W.VecData(numLocked), height, P.VecData(numLocked), height,
		     0.0, &BMat(width, width+numActive), lda );



	  // Compute P'*P
	  blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
		      P.VecData(numLocked), height, P.VecData(numLocked), height,
		      0.0, &BMat(width+numActive, width+numActive), lda );


	}

#if ( _DEBUGlevel_ >= 2 )
	{
	  DblNumMat WTW( width, width );
	  lapack::Lacpy( 'A', width, width, &BMat(width, width), lda,
			 WTW.Data(), width );


	  statusOFS << "W'*W = " << WTW << std::endl;
	  if( numSet == 3 )
	    {
	      DblNumMat PTP( width, width );
	      lapack::Lacpy( 'A', width, width, &BMat(width+numActive, width+numActive), 
			     lda, PTP.Data(), width );

	      statusOFS << "P'*P = " << PTP << std::endl;
	    }
	}
#endif



	// Rayleigh-Ritz procedure
	// AMat * C = BMat * C * Lambda
	// Assuming the dimension (needed) for C is width * width, then
	//     ( C_X )
	//     ( --- )
	// C = ( C_W )
	//     ( --- )
	//     ( C_P )
	//
	Int numCol;
	if( numSet == 3 ){
	  // Conjugate gradient
	  numCol = width + 2 * numActiveTotal;
	}
	else{
	  numCol = width + numActiveTotal;
	}

	// Solve the generalized eigenvalue problem with thresholding
   
	// Symmetrize A and B first.  This is important.
	for( Int j = 0; j < numCol; j++ ){
	  for( Int i = j+1; i < numCol; i++ ){
	    AMat(i,j) = AMat(j,i);
	    BMat(i,j) = BMat(j,i);
	  }
	}

	lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );
       
	Int numKeep = 0;
	for( Int i = numCol-1; i>=0; i-- ){
	  if( sigma2(i) / sigma2(numCol-1) >  1e-12 )
	    numKeep++;
	  else
	    break;
	}

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << "sigma2 = " << sigma2 << std::endl;
#endif

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << "sigma2(0)        = " << sigma2(0) << std::endl;
	statusOFS << "sigma2(numCol-1) = " << sigma2(numCol-1) << std::endl;
	statusOFS << "numKeep          = " << numKeep << std::endl;
#endif

	for( Int i = 0; i < numKeep; i++ ){
	  invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
	}

	if( numKeep < width ){
	  std::ostringstream msg;
	  msg 
	    << "width   = " << width << std::endl
	    << "numKeep =  " << numKeep << std::endl
	    << "there are not enough number of columns." << std::endl;
	  ErrorHandling( msg.str().c_str() );
	}

	SetValue( AMatT1, 0.0 );
	// Evaluate S^{-1/2} (U^T A U) S^{-1/2}
	blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
		    AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
		    0.0, AMatT1.Data(), lda );
  
	blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
		    BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
		    0.0, AMat.Data(), lda );
       
	for( Int j = 0; j < numKeep; j++ ){
	  for( Int i = 0; i < numKeep; i++ ){
	    AMat(i,j) *= invsigma(i)*invsigma(j);
	  }
	}
       
	// Solve the standard eigenvalue problem
	lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda,
		       eigValS.Data() );
       
	// Compute the correct eigenvectors and save them in AMat
	for( Int j = 0; j < numKeep; j++ ){
	  for( Int i = 0; i < numKeep; i++ ){
	    AMat(i,j) *= invsigma(i);
	  }
	}

	blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
		    BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
		    0.0, AMatT1.Data(), lda );

	lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
		       AMat.Data(), lda );

   

	if( numSet == 2 ){
	  // Update the eigenvectors 
	  // X <- X * C_X + W * C_W
	  blas::Gemm( 'N', 'N', height, width, width, 1.0,
		      X.Data(), height, &AMat(0,0), lda,
		      0.0, Xtemp.Data(), height );

     


	  blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
		      W.VecData(numLocked), height, &AMat(width,0), lda,
		      1.0, Xtemp.Data(), height );

	  // Save the result into X
	  lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
			 X.Data(), height );

	  // P <- W
	  lapack::Lacpy( 'A', height, numActive, W.VecData(numLocked), 
			 height, P.VecData(numLocked), height );

    
	}
	else
	  { 
	    //numSet == 3
	    // Compute the conjugate direction
	    // P <- W * C_W + P * C_P
	    blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
			W.VecData(numLocked), height, &AMat(width, 0), lda, 
			0.0, Xtemp.Data(), height );



	    blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
			P.VecData(numLocked), height, &AMat(width+numActive,0), lda,
			1.0, Xtemp.Data(), height );


	    lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
			   height, P.VecData(numLocked), height );

	    // Update the eigenvectors
	    // X <- X * C_X + P
	    blas::Gemm( 'N', 'N', height, width, width, 1.0, 
			X.Data(), height, &AMat(0,0), lda, 
			1.0, Xtemp.Data(), height );


	    lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
			   X.Data(), height );

	  } // if ( numSet == 2 ) else...


	// Update AX and AP
	if( numSet == 2 ){
	  // AX <- AX * C_X + AW * C_W
	  blas::Gemm( 'N', 'N', height, width, width, 1.0,
		      AX.Data(), height, &AMat(0,0), lda,
		      0.0, Xtemp.Data(), height );



	  blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
		      AW.VecData(numLocked), height, &AMat(width,0), lda,
		      1.0, Xtemp.Data(), height );


	  lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
			 AX.Data(), height );


	  // AP <- AW
	  lapack::Lacpy( 'A', height, numActive, AW.VecData(numLocked), height,
			 AP.VecData(numLocked), height );

	}
	else
	  {
       
	    // AP <- AW * C_W + A_P * C_P
	    blas::Gemm( 'N', 'N', height, width, numActive, 1.0, 
			AW.VecData(numLocked), height, &AMat(width,0), lda,
			0.0, Xtemp.Data(), height );



	    blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
			AP.VecData(numLocked), height, &AMat(width+numActive, 0), lda,
			1.0, Xtemp.Data(), height );



	    lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
			   height, AP.VecData(numLocked), height );

 
	    // AX <- AX * C_X + AP
	    blas::Gemm( 'N', 'N', height, width, width, 1.0,
			AX.Data(), height, &AMat(0,0), lda,
			1.0, Xtemp.Data(), height );

     
	    lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
			   AX.Data(), height );

	  } // if ( numSet == 2 ) else ...



#if ( _DEBUGlevel_ >= 1 )
	statusOFS << "numLocked = " << numLocked << std::endl;
	statusOFS << "eigValS   = " << eigValS << std::endl;
#endif



      } // for (iter) end for main loop




    // *********************************************************************
    // Post processing
    // *********************************************************************

    // Obtain the eigenvalues and eigenvectors
    // XTX should now contain the matrix X' * (AX), and X is an
    // orthonormal set
    lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );


    // X <- X*C
    blas::Gemm( 'N', 'N', height, width, width, 1.0, X.Data(),
		height, XTX.Data(), width, 0.0, Xtemp.Data(), height );

 

    lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
		   X.Data(), height );


#if ( _DEBUGlevel_ >= 2 )

    blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
		height, X.Data(), height, 0.0, XTX.Data(), width );


    statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;
#endif


    // Copy back results
    lapack::Lacpy( 'A', height, width, X.Data(), height, 
		   Xmat.Data(), height );
    for(Int copy_iter = 0; copy_iter < width; copy_iter ++)
      eig_vals_Xmat(copy_iter) = -eigValS(copy_iter); 
 
    //statusOFS << std::endl << " Eigvals = " << std::endl <<  eigValS;
    //statusOFS << std::endl << " Res = " << std::endl <<  resNorm; 

    if( isConverged ){
      statusOFS << std::endl << " After " << iter 
		<< " iterations, LOBPCG has converged."  << std::endl
		<< " The maximum norm of the residual is " 
		<< resMax << std::endl << std::endl;
    }
    else{
      statusOFS << std::endl << " After " << iter 
		<< " iterations, LOBPCG did not converge. " << std::endl
		<< " The maximum norm of the residual is " 
		<< resMax << std::endl 
		<< " Desired max residual = " << eigTolerance << std::endl
	        << std::endl;

     
    }


    return ;
  }     

}

#endif
