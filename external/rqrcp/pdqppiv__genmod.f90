        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE PDQPPIV__genmod
          INTERFACE 
            SUBROUTINE PDQPPIV(M,N,A,IA,JA,DESCA,IPIV)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: IA
              INTEGER(KIND=4) :: JA
              INTEGER(KIND=4) :: DESCA(*)
              INTEGER(KIND=4) :: IPIV(*)
            END SUBROUTINE PDQPPIV
          END INTERFACE 
        END MODULE PDQPPIV__genmod
