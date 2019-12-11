        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE PDGEQP3S__genmod
          INTERFACE 
            SUBROUTINE PDGEQP3S(M,N,OFFSET,A,DESCA,G,DESCG,VN1,VN2,DESCN&
     &,IPIV,TAU,NBIN,NBOUT,DWORK,IWORK)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              INTEGER(KIND=4) :: OFFSET
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: DESCA(*)
              REAL(KIND=8) :: G(*)
              INTEGER(KIND=4) :: DESCG(*)
              REAL(KIND=8) :: VN1(*)
              REAL(KIND=8) :: VN2(*)
              INTEGER(KIND=4) :: DESCN(*)
              INTEGER(KIND=4) :: IPIV(*)
              REAL(KIND=8) :: TAU(*)
              INTEGER(KIND=4) :: NBIN
              INTEGER(KIND=4) :: NBOUT
              REAL(KIND=8) :: DWORK(*)
              INTEGER(KIND=4) :: IWORK(*)
            END SUBROUTINE PDGEQP3S
          END INTERFACE 
        END MODULE PDGEQP3S__genmod
