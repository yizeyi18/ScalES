        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE CHECK_QPF__genmod
          INTERFACE 
            FUNCTION CHECK_QPF(M,N,K,A,DESCA,B,DESCB,C,DESCC,TAU,IPIV,  &
     &WORK,LWORK)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              INTEGER(KIND=4) :: K
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: DESCA(*)
              REAL(KIND=8) :: B(*)
              INTEGER(KIND=4) :: DESCB(*)
              REAL(KIND=8) :: C(*)
              INTEGER(KIND=4) :: DESCC(*)
              REAL(KIND=8) :: TAU(*)
              INTEGER(KIND=4) :: IPIV(*)
              REAL(KIND=8) :: WORK(*)
              INTEGER(KIND=4) :: LWORK
              REAL(KIND=8) :: CHECK_QPF
            END FUNCTION CHECK_QPF
          END INTERFACE 
        END MODULE CHECK_QPF__genmod
