        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:09:54 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE LB1__genmod
          INTERFACE 
            SUBROUTINE LB1(IPRINT,ITER,NFUN,GNORM,N,M,X,F,G,STP,FINISH)
              INTEGER(KIND=4) :: N
              INTEGER(KIND=4) :: IPRINT(2)
              INTEGER(KIND=4) :: ITER
              INTEGER(KIND=4) :: NFUN
              REAL(KIND=8) :: GNORM
              INTEGER(KIND=4) :: M
              REAL(KIND=8) :: X(N)
              REAL(KIND=8) :: F
              REAL(KIND=8) :: G(N)
              REAL(KIND=8) :: STP
              LOGICAL(KIND=4) :: FINISH
            END SUBROUTINE LB1
          END INTERFACE 
        END MODULE LB1__genmod
