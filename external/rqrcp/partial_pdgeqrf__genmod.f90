        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE PARTIAL_PDGEQRF__genmod
          INTERFACE 
            SUBROUTINE PARTIAL_PDGEQRF(M,N,K,A,IA,JA,DESCA,TAU,WORK,    &
     &LWORK,INFO)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              INTEGER(KIND=4) :: K
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: IA
              INTEGER(KIND=4) :: JA
              INTEGER(KIND=4) :: DESCA(*)
              REAL(KIND=8) :: TAU(*)
              REAL(KIND=8) :: WORK(*)
              INTEGER(KIND=4) :: LWORK
              INTEGER(KIND=4) :: INFO
            END SUBROUTINE PARTIAL_PDGEQRF
          END INTERFACE 
        END MODULE PARTIAL_PDGEQRF__genmod
