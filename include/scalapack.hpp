/// @file scalapack.hpp
/// @brief Thin interface to ScaLAPACK
/// @author Lin Lin
/// @date 2012-06-05
#ifndef _SCALAPACK_HPP_
#define _SCALAPACK_HPP_

#include  "environment.hpp"

namespace dgdft{
namespace scalapack{

typedef  int                    Int;
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;


// *********************************************************************
// Cblas  routines
// *********************************************************************
extern "C"{

void Cblacs_get(const Int contxt, const Int what, Int* val);

void Cblacs_gridinit(Int* contxt, const char* order, const Int nprow, const Int npcol);

void Cblacs_gridmap(Int* contxt, Int* pmap, const Int ldpmap, const Int nprow, const Int npcol);

void Cblacs_gridinfo(const Int contxt,  Int* nprow, Int* npcol, 
		Int* myprow, Int* mypcol);

void Cblacs_gridexit	(	int contxt );	

}

// *********************************************************************
// Interface for Descriptor
// *********************************************************************
class Descriptor{
private:
	std::vector<Int> values_;
	Int nprow_;
	Int npcol_;
	Int myprow_;
	Int mypcol_;
public:
	//in total 9 elements in the descriptor
	//NOTE: C convention is followed here!
	enum{
		DTYPE = 0,
		CTXT  = 1,
		M     = 2,
		N     = 3,
		MB    = 4,
		NB    = 5,
		RSRC  = 6,
		CSRC  = 7,
		LLD   = 8,
		DLEN  = 9, 
	};

	Descriptor() {values_.resize(DLEN);}

	/// @brief Constructor.
	///
	/// NOTE: The leading dimension is directly computed rather than
	/// taking as an input.
	Descriptor(Int m, Int n, Int mb,
			Int nb, Int irsrc, Int icsrc,
			Int contxt) 
	{Init(m, n, mb, nb, irsrc, icsrc, contxt);}

	~Descriptor(){}

	/// @brief Initialize the descriptor. 
	void Init(Int m, Int n, Int mb,
			Int nb, Int irsrc, Int icsrc,
			Int contxt);
	
	Int* Values() {return &values_[0];} 
	
	const Int* Values() const {return &values_[0];}

	Int NpRow() const {return nprow_;}

	Int NpCol() const {return npcol_;}

	Int MypRow() const {return myprow_;}

	Int MypCol() const {return mypcol_;}

	Int Get(Int i) const
	{
		if( i < 0 || i > DLEN ){
			std::ostringstream msg;
			msg << "Descriptor::Get takes value in [0,8]" << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
		return values_[i];
	}


	Descriptor& operator=(const Descriptor& desc);
};


/*********************************************************************************
 * Interface for a ScaLAPACK Matrix
 *********************************************************************************/

template<typename F>
class ScaLAPACKMatrix{
private:
	Descriptor       desc_;
	std::vector<F>   localMatrix_;
public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
	ScaLAPACKMatrix(){}

	~ScaLAPACKMatrix(){}

	ScaLAPACKMatrix<F>& operator=(const ScaLAPACKMatrix<F>& A);


	/************************************************************
	 * Basic information
	 ************************************************************/

	Int Height() const {return desc_.Get(Descriptor::M);}

	Int Width() const {return desc_.Get(Descriptor::N);}

	Int MB()    const {return desc_.Get(Descriptor::MB);}

	Int NB()    const {return desc_.Get(Descriptor::NB);}

	Int Context() const {return desc_.Get(Descriptor::CTXT);}

	Int NumRowBlocks() const 
	{return (this->Height() + this->MB() - 1) / this->MB();}

	Int NumColBlocks() const
	{return (this->Width()  + this->NB() - 1) / this->NB();}

	// NOTE: LocalHeight is the same as LocalLDim here.
	Int LocalNumRowBlocks() const
	{ return (this->NumRowBlocks() + this->desc_.NpRow() - 1 ) /
		this->desc_.NpRow(); }

	Int LocalNumColBlocks() const
	{return (this->NumColBlocks() + this->desc_.NpCol() - 1 ) /
		this->desc_.NpCol(); }

	Int LocalHeight() const 
	{return this->LocalNumRowBlocks() * this->MB(); }

	Int LocalWidth() const 
	{return this->LocalNumColBlocks() * this->NB(); }

	Int LocalLDim()  const 
	{ 
		if(desc_.Get(Descriptor::LLD) != this->LocalHeight() );
		{
			std::ostringstream msg;
			msg << "ScaLAPACK: the leading dimension does not match" << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
		return desc_.Get(Descriptor::LLD);
	}

	class Descriptor& Desc() {return desc_;}

	const class Descriptor&  Desc() const {return desc_;}

	F*  Data() {return &localMatrix_[0];}

	const F* Data() const {return &localMatrix_[0];}

	std::vector<F>&  LocalMatrix() {return localMatrix_;}

	/// @brief Change the descriptor of the matrix.
	///
	/// NOTE: Changing the descriptor is the only way to resize the
	/// localMatrix.
	void SetDescriptor(const class Descriptor& desc)
	{
		desc_ = desc;
		localMatrix_.resize(this->LocalHeight() * this->LocalWidth());
	}

	/************************************************************
	 * Entry manipulation
	 ************************************************************/
	F  GetLocal( Int iLocal, Int jLocal ) const;

	void SetLocal( Int iLocal, Int jLocal, F val );
};


/*********************************************************************************
 * Methods
 *********************************************************************************/

/// @brief Redistribute a ScaLAPACKMatrix A into a ScaLAPACKMatrix
/// B which shares the same context as A. 
///
/// Performs p_gemr2d.
void
Gemr2d(const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B);


/// @brief Compute the eigenvalues only for symmetric matrices. 
///
/// Performs p_syev.  
void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
		std::vector<double>& eigs);

/// @brief Compute the eigenvalues and the eigenvectors for symmetric
/// matrices.  
///
/// Performs p_syev.  
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
		std::vector<double>& eigs,
		ScaLAPACKMatrix<double>& Z);

/// @brief Compute the eigenvalues and the eigenvectors for symmetric
/// matrices using the divide-and-conquer algorithm.  
///
/// Performs p_syevd.
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
void
Syevd(char uplo, ScaLAPACKMatrix<double>& A, 
		std::vector<double>& eigs,
		ScaLAPACKMatrix<double>& Z);

/// @brief Compute the eigenvalues and the eigenvectors for symmetric
/// matrices using the MRRR algoritm for diagonalizing the tri-diagonal
/// problem.
///
/// Performs p_syevd.
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
void 
Syevr(char uplo, ScaLAPACKMatrix<double>& A,
		std::vector<double>& eigs,
		ScaLAPACKMatrix<double>& Z);


} // namespace scalapack
} // namespace dgdft

#endif // _SCALAPACK_HPP_
