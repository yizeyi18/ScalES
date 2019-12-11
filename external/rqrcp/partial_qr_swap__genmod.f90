        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE PARTIAL_QR_SWAP__genmod
          INTERFACE 
            SUBROUTINE PARTIAL_QR_SWAP(PARTIAL,M,N,A,IA,JA,DESCA,IPIV,  &
     &TAU,WORK,LWORK,INFO,SWAP_M,SWAP_LDA,SWAP_A,SWAP_DESCA,SWAP_IPIV)
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
              INTEGER(KIND=4) :: SWAP_M
              INTEGER(KIND=4) :: SWAP_LDA
              REAL(KIND=8) :: SWAP_A(*)
              INTEGER(KIND=4) :: SWAP_DESCA(*)
              INTEGER(KIND=4) :: SWAP_IPIV(*)
            END SUBROUTINE PARTIAL_QR_SWAP
          END INTERFACE 
        END MODULE PARTIAL_QR_SWAP__genmod
