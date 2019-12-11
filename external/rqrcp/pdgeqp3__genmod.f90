        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:10:11 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE PDGEQP3__genmod
          INTERFACE 
            SUBROUTINE PDGEQP3(M,N,A,DESCA,IPIV,TAU,DWORK,LDWORK,IWORK, &
     &LIWORK,INFO)
              INTEGER(KIND=4) :: M
              INTEGER(KIND=4) :: N
              REAL(KIND=8) :: A(*)
              INTEGER(KIND=4) :: DESCA(*)
              INTEGER(KIND=4) :: IPIV(*)
              REAL(KIND=8) :: TAU(*)
              REAL(KIND=8) :: DWORK(*)
              INTEGER(KIND=4) :: LDWORK
              INTEGER(KIND=4) :: IWORK(*)
              INTEGER(KIND=4) :: LIWORK
              INTEGER(KIND=4) :: INFO
            END SUBROUTINE PDGEQP3
          END INTERFACE 
        END MODULE PDGEQP3__genmod
