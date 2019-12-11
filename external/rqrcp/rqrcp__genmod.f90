        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE RQRCP__genmod
          INTERFACE 
            SUBROUTINE RQRCP(M,N,K,A,DESC_A,M_B,N_B,B,DESC_B,OMEGA,     &
     &DESC_OMEGA,IPIV,TAU,NB_MAX,IPIV_B,TAU_B,WORK,LWORK)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              INTEGER(KIND=4) :: K
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: DESC_A(*)
              INTEGER(KIND=4) :: M_B
              INTEGER(KIND=4) :: N_B
              REAL(KIND=8) :: B(*)
              INTEGER(KIND=4) :: DESC_B(*)
              REAL(KIND=8) :: OMEGA(*)
              INTEGER(KIND=4) :: DESC_OMEGA(*)
              INTEGER(KIND=4) :: IPIV(*)
              REAL(KIND=8) :: TAU(*)
              INTEGER(KIND=4) :: NB_MAX
              INTEGER(KIND=4) :: IPIV_B(*)
              REAL(KIND=8) :: TAU_B(*)
              REAL(KIND=8) :: WORK(*)
              INTEGER(KIND=4) :: LWORK
            END SUBROUTINE RQRCP
          END INTERFACE 
        END MODULE RQRCP__genmod
