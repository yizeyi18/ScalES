/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin

   This file is part of DGDFT. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   (1) Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
   (2) Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
   (3) Neither the name of the University of California, Lawrence Berkeley
   National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You are under no obligation whatsoever to provide any bug fixes, patches, or
   upgrades to the features, functionality or performance of the source code
   ("Enhancements") to anyone; however, if you choose to make your Enhancements
   available either publicly, or directly to Lawrence Berkeley National
   Laboratory, without imposing a separate written license agreement for such
   Enhancements, then you hereby grant the following license: a non-exclusive,
   royalty-free perpetual license to install, use, modify, prepare derivative
   works, incorporate into other computer software, distribute, and sublicense
   such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file iondynamics.hpp
/// @brief Geometry optimization and molecular dynamics for ions
/// @date 2015-03-05 Organize previously implemented methods
#ifndef _IONDYNAMICS_HPP_
#define _IONDYNAMICS_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace dgdft{

enum NLCG_call_type{NLCG_INIT, NLCG_CALL_TYPE_1, NLCG_CALL_TYPE_2, NLCG_CALL_TYPE_3, NLCG_CALL_TYPE_4};   

class NLCG_internal_vars_type
{
private:



public:

    // User defined variables : read from esdf or use defaults
    int i_max_;
    int j_max_;
    int n_;
    double epsilon_tol_outer_;
    double epsilon_tol_inner_;
    double sigma_0_;

    NLCG_call_type call_type;
    int numAtom;

    int i_;
    int j_;
    int k_;

    std::vector<Point3>  atompos_x_;
    std::vector<Point3>  atomforce_r_;
    std::vector<Point3>  atomforce_s_;
    std::vector<Point3>  atomforce_d_;

    double delta_new_;
    double delta_mid_;
    double delta_old_;

    double delta_0_;
    double delta_d_;

    double eta_prev_;
    double eta_;
    double alpha_;
    double beta_;

    // Initialization routine
    void setup(int i_max, int j_max, int n, 
            double epsilon_tol_outer, double epsilon_tol_inner, double sigma_0,
            std::vector<Atom>&   atomList)
    {
        call_type = NLCG_INIT;

        i_max_ = i_max;
        j_max_ = j_max;
        n_ = n;
        epsilon_tol_outer_ = epsilon_tol_outer;
        epsilon_tol_inner_ = epsilon_tol_inner;
        sigma_0_ = sigma_0;

        // Prepare the position and force lists
        numAtom = atomList.size();
        atompos_x_.resize(numAtom);
        atomforce_r_.resize(numAtom);
        atomforce_s_.resize(numAtom);
        atomforce_d_.resize(numAtom);



        i_ = 0;
        j_ = 0;
        k_ = 0;

        // Set r = -f'(x) : Note that  atomList[a].force = - grad E already - so no extra negative required
        for( Int a = 0; a < numAtom; a++ )
            atomforce_r_[a] = atomList[a].force;

        // Also copy the atom positions
        for( Int a = 0; a < numAtom; a++ )
            atompos_x_[a] = atomList[a].pos;


        // Set s = M^{-1} r : M = Identity used here
        for( Int a = 0; a < numAtom; a++ )
            atomforce_s_[a] = atomforce_r_[a];

        // Set d = s
        for( Int a = 0; a < numAtom; a++ )
            atomforce_d_[a] = atomforce_s_[a];

        // Set delta_new = r^T d  
        delta_new_ = atom_ddot(atomforce_r_, atomforce_d_);

        // Set delta_0 = delta_new
        delta_0_ = delta_new_;



        call_type = NLCG_CALL_TYPE_1;

        return;

    }

    // Computes the dot product 
    double atom_ddot(std::vector<Point3>&  list_1, std::vector <Point3>&  list_2)
    {

        double ans = 0.0;

        for( Int a = 0; a < numAtom; a++ )
        {
            for( Int d = 0; d < DIM; d++ )
            {
                ans +=  (list_1[a][d] * list_2[a][d]);
            }
        }





        return ans;    

    }


};


class IonDynamics
{
private:
    /// @brief Syncec with atomic position (and velocity and force) from
    /// input. Currently atomList info is stored in the Hamiltonian
    /// classes (Hamiltonian or HamiltonianDG).
    std::vector<Atom>*   atomListPtr_;

    /// @brief Max number of history for density extrapolation etc
    Int maxHist_;

    /// @brief atomListHist_[0] stores the lastest atomic configuration
    std::vector<std::vector<Atom> >   atomListHist_;

    std::string          ionMove_;

    DblNumVec            atomMass_;

    // Taken from input
    bool                 isOutputPosition_; 
    bool                 isOutputVelocity_;
    bool                 isOutputXYZ_;
    std::string          MDExtrapolationType_;

    // Molecular dynamics variables
    Real                 Ekinetic_; // kinetic energy for ions
    Real                 Epot_;  // potential energy for ions
    Real                 EconserveInit_;
    Real                 Econserve_;
    Real                 Edrift_;
    Real                 dt_;
    Real                 ionTemperature_; // unit: au

    // Nose-Hoover variables
    Real                 Q1_;
    Real                 xi1_;
    Real                 vxi1_;
    Real                 G1_;
    Real                 scalefac_;
    Int                  phase_; // NH1 has two phases
    // Langevin variables
    Real                 langevinDamping_;

    bool                 isGeoOpt_;
    bool                 isMD_;


    /// @brief BarzilaiBorwein method for geometry optimization
    ///
    void BarzilaiBorweinOpt( Int ionIter );

    /// @brief Non-linear Conjugate Gradient with Secant and Polak-Ribiere
    NLCG_internal_vars_type NLCG_vars;
    void NLCG_Opt(Int ionIter );

    //  void FireOpt( Int ionIter );

    /// @brief VelocityVerlet for NVE simulation
    ///
    void VelocityVerlet( Int ionIter );

    /// @brief NoseHoover thermostat with chain level 1. The
    /// implementation is consistent with the CORRECT version of
    ///
    /// Frenkel and Smit, Understanding Molecular Simulation, 
    /// Alg. 30 (pp 540)
    ///
    /// The correction is due to the inconsistency between (E.2.4) and the
    /// pos_vel algorithm (Alg. 32). The algorithm in the book is correct,
    /// however, it does not allow one to obtain position and velocity at
    /// the same snapshot.
    /// 
    /// Normally one NH step is (3.2.4)
    ///
    ///   *Time t
    ///   chain (dt/2)
    ///   velocity (dt/2)
    ///   position (dt)
    ///   evaluate force
    ///   velocity (dt/2)
    ///   chain (dt/2)
    ///   *Time t+dt
    ///
    /// Since we do not want to call MoveIons twice, the order is switched
    /// to the following
    ///
    ///   evaluate force
    ///   velocity (dt/2)
    ///   chain (dt/2)
    ///   *Time t+dt
    ///   chain (dt/2)
    ///   velocity (dt/2)
    ///   position (dt)
    ///
    /// This means that after updating the chain variables for the first
    /// time, the position, velocity, energy and forces are synced at time
    /// t+dt, and then chain, velocity and position variables are updated
    /// in preparation for evaluation of the force.  
    ///
    /// If the job is stopped in the middle of the evaluation of the
    /// force, the calculation can be restarted from the stored variables
    /// without repeating the work. (3/6/2016)
    /// 
    void NoseHoover1( Int ionIter );

    // Integrator 
    /// @brief Langevin integrator
    /// Reference 
    ///
    /// N. Gronbech-Jensen, O. Farago, A simple and effective Verlet-type
    /// algorithm for simulating Langevin dynamics, Mol.  Phys.  111
    /// (2013) 983–991
    void Langevin( Int ionIter );



public:

    /// @brief Initial setup from the input parameters
    void Setup( const esdf::ESDFInputParam& esdfParam, std::vector<Atom>& atomList,
            PeriodTable& ptable );

    /// @brief Main program to move the ions.
    ///
    /// Will determine both geometry optimization and molecular dynamics
    void MoveIons( Int ionIter );

    /// @brief Extrapolating coefficient for density or wavefunction
    ///
    void ExtrapolateCoefficient( Int ionIter, DblNumVec& coef );

    // *********************************************************************
    // Access functions
    // *********************************************************************
    std::vector<Atom>& AtomList() { return *atomListPtr_; }
    void SetEpot(Real Epot) { Epot_ = Epot; }
    Int  MaxHist() { return maxHist_; }

    bool IsGeoOpt() { return isGeoOpt_; }
    bool IsMD()     { return isMD_; }


};


} // namespace dgdft


#endif // _IONDYNAMICS_HPP_
