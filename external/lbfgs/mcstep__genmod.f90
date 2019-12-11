        !COMPILER-GENERATED INTERFACE MODULE: Thu Apr 18 01:09:54 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE MCSTEP__genmod
          INTERFACE 
            SUBROUTINE MCSTEP(STX,FX,DX,STY,FY,DY,STP,FP,DP,BRACKT,     &
     &STPMIN,STPMAX,INFO)
              REAL(KIND=8) :: STX
              REAL(KIND=8) :: FX
              REAL(KIND=8) :: DX
              REAL(KIND=8) :: STY
              REAL(KIND=8) :: FY
              REAL(KIND=8) :: DY
              REAL(KIND=8) :: STP
              REAL(KIND=8) :: FP
              REAL(KIND=8) :: DP
              LOGICAL(KIND=4) :: BRACKT
              REAL(KIND=8) :: STPMIN
              REAL(KIND=8) :: STPMAX
              INTEGER(KIND=4) :: INFO
            END SUBROUTINE MCSTEP
          END INTERFACE 
        END MODULE MCSTEP__genmod
