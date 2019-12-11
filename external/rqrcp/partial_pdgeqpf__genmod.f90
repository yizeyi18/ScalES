        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE PARTIAL_PDGEQPF__genmod
          INTERFACE 
            SUBROUTINE PARTIAL_PDGEQPF(PARTIAL,M,N,A,IA,JA,DESCA,IPIV,  &
     &TAU,WORK,LWORK,INFO)
              INTEGER(KIND=4) :: PARTIAL
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: IA
              INTEGER(KIND=4) :: JA
              INTEGER(KIND=4) :: DESCA(*)
              INTEGER(KIND=4) :: IPIV(*)
              REAL(KIND=8) :: TAU(*)
              REAL(KIND=8) :: WORK(*)
              INTEGER(KIND=4) :: LWORK
              INTEGER(KIND=4) :: INFO
            END SUBROUTINE PARTIAL_PDGEQPF
          END INTERFACE 
        END MODULE PARTIAL_PDGEQPF__genmod
