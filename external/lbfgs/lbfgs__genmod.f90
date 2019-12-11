        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:09:54 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE LBFGS__genmod
          INTERFACE 
            SUBROUTINE LBFGS(N,M,X,F,G,DIAGCO,DIAG,IPRINT,EPS,XTOL,W,   &
     &IFLAG)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              REAL(KIND=8) :: X(N)
              REAL(KIND=8) :: F
              REAL(KIND=8) :: G(N)
              LOGICAL(KIND=4) :: DIAGCO
              REAL(KIND=8) :: DIAG(N)
              INTEGER(KIND=4) :: IPRINT(2)
              REAL(KIND=8) :: EPS
              REAL(KIND=8) :: XTOL
              REAL(KIND=8) :: W(N*(2*M+1)+2*M)
              INTEGER(KIND=4) :: IFLAG
            END SUBROUTINE LBFGS
          END INTERFACE 
        END MODULE LBFGS__genmod
