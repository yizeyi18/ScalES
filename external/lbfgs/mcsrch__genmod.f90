        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:09:54 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE MCSRCH__genmod
          INTERFACE 
            SUBROUTINE MCSRCH(N,X,F,G,S,STP,FTOL,XTOL,MAXFEV,INFO,NFEV, &
     &WA)
              INTEGER(KIND=4) :: N
              REAL(KIND=8) :: X(N)
              REAL(KIND=8) :: F
              REAL(KIND=8) :: G(N)
              REAL(KIND=8) :: S(N)
              REAL(KIND=8) :: STP
              REAL(KIND=8) :: FTOL
              REAL(KIND=8) :: XTOL
              INTEGER(KIND=4) :: MAXFEV
              INTEGER(KIND=4) :: INFO
              INTEGER(KIND=4) :: NFEV
              REAL(KIND=8) :: WA(N)
            END SUBROUTINE MCSRCH
          END INTERFACE 
        END MODULE MCSRCH__genmod
