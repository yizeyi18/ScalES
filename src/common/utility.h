//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Weile Jia

/// @file utility.hpp
/// @brief Utility subroutines.
/// @date 2012-08-12
#ifndef _UTILITY_HPP_ 
#define _UTILITY_HPP_

#include  <stdlib.h>
#include  "domain.h"
#include  "environment.h"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"
#include  "nummat_impl.hpp"
#include  "numtns_impl.hpp"
#include  "sparse_matrix_impl.hpp"
#ifdef GPU
#include  "cu_nummat_impl.hpp"
#include  "cu_numvec_impl.hpp"
#include  "cuda_utils.h"
#endif
namespace scales{

#ifdef _PROFILING_ //计时
extern Real alltoallTime;
extern Real alltoallTimeTotal ;
void reset_alltoall_time();
#endif


// Forward declaration of Atom structure in periodtable.h 
struct Atom;

// *********************************************************************
// Global utility functions 
// These utility functions do not depend on local definitions
// *********************************************************************
inline Int IRound(Real a){                                     //四舍五入 
  Int b = 0;
  if(a>0) b = (a-Int(a)<0.5)?Int(a):(Int(a)+1);
  else b = (Int(a)-a<0.5)?Int(a):(Int(a)-1);
  return b; 
}

inline Int IMod(int a,int b){                                  //取余，返回非负数
  assert( b > 0 );//给编译器看的
  return ((a%b)<0)?((a%b)+b):(a%b);
}

inline Real DMod( Real a, Real b){                             //浮点取“余”，返回非负浮点a-nb
  assert(b>0);
  return (a>=0)?(a-Int(a/b)*b):(a-(Int(a/b)-1)*b);
}


inline Int OptionsCreate(Int argc, char** argv, std::map<std::string,std::string>& options)
{                                                              //解释命令行参数？options存储结果
  options.clear();
  for(Int k=1; k<argc; k=k+2) {//argv[0]是可执行文件绝对路径，假设每个argv都以k+1为值
    options[ std::string(argv[k]) ] = std::string(argv[k+1]);
  }
  return 0;
}

template<class Element, class Container>
bool InArray(const Element & element, const Container & container)
{                                                              //查找元素是否在容器里
  return std::find(std::begin(container), std::end(container), element)
    != std::end(container);
}


// *********************************************************************
// Stringstream 字符流
// *********************************************************************

// Size information in byte 以字节为单位的字符流长度
// Like sstm.str().length() but without making the copy
inline Int Size( std::stringstream& sstm ){
  Int length;
  sstm.seekg (0, std::ios::end);
  length = sstm.tellg();
  sstm.seekg (0, std::ios::beg);//设回起点
  return length;
}


// *********************************************************************
// Data types 数据类型
// *********************************************************************
// Sparse vector used for a domain with uniform grid (not for LGL grid).  
// SparseVec.first is the indices of grid points for which the sparse
// vector has nonzero value.  
// SparseVec.second contains four columns, ordered as value, and its
// derivatives along x, y, z directions, respectively. 
typedef std::pair<IntNumVec, DblNumMat > SparseVec; //稀疏向量
						    //存储格点上的值与xyz方向上的导数
						    //为啥不直接用个5*n矩阵？
// Each nonlocal pseudopotential is written as
//   \int dy V(x,y) f(y) = w b(x) \int dy b(y) f(y).
//   其中x、y、V、b的含义不明确。
//   x、y可能分别是三维坐标
//   V是非局域势函数
//   f是波函数？
//   b是势函数分离变量版？
//   以及，非局域势求起来应该是<psi|V|psi>
//   此处做了分离变量假设？
//   FIXME
// First: b vector with integration measure.
// Second: weight w
typedef std::pair<SparseVec, Real> NonlocalPP;      //非局域赝势
						    //存储一种非局域势的权重及其在格点上的值
/// @struct PseudoPot
/// @brief The pseudocharge and nonlocal projectors for each atom. 
///
/// Each vector is on the global grid in the format of SparseVec.
struct PseudoPot                                    //赝势结构体
{
  /// @brief Pseudocharge of an atom, defined on the uniform fine grid.
  /// When VLocal is present, pseudoCharge corresponds to the Gaussian
  /// compensation charge
  SparseVec                         pseudoCharge;   //格点电荷
  /// @brief Short range local potential of an atom, defined on the uniform fine grid.
  SparseVec                         vLocalSR;       //格点上的短程局域势
  /// @brief Nonlocal projectors of an atom, defined on the uniform fine grid.
  std::vector<NonlocalPP>           vnlList;        //格点上的非局域势
						    //可能不止一种所以用了vector
};




// *********************************************************************
// Mesh  and other utilities FIXME //网络及其它。为什么这玩意会在头文件里？
// *********************************************************************

/// @brief Check whether a point is in a sub-domain which is a part of a
/// global domain with periodic boundary conditions
inline bool IsInSubdomain(                          //检查r是否在dm里
    const Point3& r, 
    const Domain& dm, 
    const Point3& Lsglb )                           //周期性条件/晶胞参数
{
  bool isIn = true;
  Point3 posstart = dm.posStart;                    //dm原点在global坐标系下的坐标
  Point3 Lsbuf = dm.length;                         //dm在三个方向上的长度
  Point3 shiftstart;
  Point3 shiftr;
  for( Int i = 0; i < DIM; i++){                    //DIM就是3，于common/environment.h定义
    shiftstart[i] = DMod(posstart[i], Lsglb[i]);    //求dm原点的分数坐标/在晶胞、周期性条件里的坐标
    shiftr[i]     = DMod(r[i],        Lsglb[i]);    //求r的分数坐标 DMod：55行定义的浮点取余
    /* Case 1 of the buffer interval */             // 图例：晶胞是| dm起止点是/ dm内点是·
    if( shiftstart[i] + Lsbuf[i] > Lsglb[i] ){      //原点坐标+长度>周期性条件，换句话说跨晶胞了
      if( (shiftr[i] < shiftstart[i]) &&            //r在原点左            |···/  r  /···| out
	  (shiftr[i] > shiftstart[i] + Lsbuf[i] - Lsglb[i]) ) //且在终点右 |···/     /·r·| in
        isIn = false;                               //                     |·r·/     /···| in
    }//if
    /* Case 2 of the buffer interval */
    else{                                           //dm在一个晶胞内 |   /··r··/   | in
      if( (shiftr[i] < shiftstart[i]) ||            //r在原点左      | r /·····/   | out
          (shiftr[i] > shiftstart[i] + Lsbuf[i]) ){ //或终点右       |   /·····/ r | out
        isIn = false;
      }
    }
  }
  return isIn;
}

/// @brief Converts a 1D global index to a 3D index.
// 将grid编号转化为三指标编号
inline Index3 Index1To3( const Int idx1, const Index3& numGrid ){//FIXME numGrid不变，用&有什么作用？
  Index3 idx3;
  idx3[2] = idx1 / ( numGrid[0] * numGrid[1] );
  idx3[1] = ( idx1 % ( numGrid[0] * numGrid[1] ) ) / numGrid[0];
  idx3[0] = idx1 % numGrid[0];
  return idx3;
}

/// @brief Converts a 3D index to a 1D global index.
// 这个指导上面那个
inline Int Index3To1( const Index3& idx3, const Index3& numGrid ){
  Int idx1 = idx3[0] + idx3[1] * numGrid[0] + idx3[2] * numGrid[0] * numGrid[1];
  return idx1;
}


/// @brief Interface for generating the unscaled LGL grid points and
/// integration weights.
///
/// Compared to GenerateLGL, the Legendre polynomials are only
/// recursively computed and not stored, and the differentiation matrix
/// is neither computed nor stored.  This allows the evaluation of a
/// very large number of LGL grids.
///
/// Note: size(x) = size(w) = size(P,1|2) = size(D,1|2) = N
//
//  生成LGL格点与积分权重的接口。
//  可以生成很大规模的LGL格点。
//  另有个叫GenerateLGL的接口，会存储勒让德多项式，
//  并计算、存储差分矩阵。吃资源比这个多。
//
//  LGL: Legendre-Gauss-Lobatto，格点名
//  见：https://doi.org/10.48550/arXiv.1311.0028
//
//  P,D,x指代暂不明确，w应该是weight，N应该是格点数
//  为什么把N单独作为参数传入而不从对象属性里读？
//  更新：P是勒让德多项式，D是差分矩阵。作用待查。
void GenerateLGLMeshWeightOnly(
    DblNumVec&         x, 
    DblNumVec&         w, 
    Int                N);

/// @brief Actual subroutine for generating the unscaled LGL grid points and
/// integration weights.
///
/// Compared to GenerateLGL, the Legendre polynomials are only
/// recursively computed and not stored, and the differentiation matrix
/// is neither computed nor stored.  This allows the evaluation of a
/// very large number of LGL grids.

/// Note: for legacy reason,  
/// size(x) = size(w) = N-1
//  ↑这里的size()是什么？FIXME
//
//  上面接口的本体，把自定义数据类型换成了原生数据类型。
void GenerateLGLMeshWeightOnly(double* x, double* w, int Nm1);


/// @brief Interface for generating the unscaled LGL grid points, integration weights, polynomials and differentiation matrix.
///
/// Note: size(x) = size(w) = size(P,1|2) = size(D,1|2) = N
//
//  生成LGL格点的接口，会计算、存储勒让德多项式与差分矩阵。
void GenerateLGL(
    DblNumVec&         x, 
    DblNumVec&         w, 
    DblNumMat&         P,
    DblNumMat&         D,
    Int                N);

/// @brief Actual subroutine for generating the unscaled LGL grid points, integration weights, polynomials and differentiation matrix.
///
/// Note: for legacy reason,  
/// size(x) = size(w) = size(P,1|2) = size(D,1|2) = N-1
//
//  上述接口的本体，把自定义数据类型换成了原生数据类型。
void GenerateLGL(double* x, double* w, double* P, double* D, int Nm1);

template<class F>//转置，但转置后的矩阵存在B里。使用std::vector。
void Transpose(std::vector<F>& A, std::vector<F>& B, Int m, Int n){
  Int i, j;
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      B[j+n*i] = A[i+j*m];
    }
  }
}

template<class F>//转置，但转置后的矩阵存在B里。使用Real，或者说double。
void Transpose(Real* A, Real* B, Int m, Int n){//有什么用模板的必要吗？
  Int i, j;
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      B[j+n*i] = A[i+j*m];
    }
  }
}

/// @brief Generate 1D spline coefficients
//
//  FIXME 生成“1D spline”系数。没看懂。
//  或许应该把注释写在.cpp里而不是头文件里。
void spline(Int, Real*, Real*, Real*, Real*, Real*);

/// @brief Evaluate the spline
//
//  生成spline。
void seval(Real*, Int, Real*, Int, Real*, Real*, Real*, Real*, Real*);


/// @brief Three component inner product.
//
//  三个向量的“内积”。
inline Real Innerprod(Real* x, Real* y, Real *w, Int ntot){
  Real tmp = 0.0;
  for(Int i = 0; i < ntot; i++){
    tmp += x[i] * y[i] * w[i];
  }
  return tmp;
}

/// @brief Generate a uniform mesh from a domain.
void UniformMesh( const Domain &dm, std::vector<DblNumVec> &gridpos );
void UniformMeshFine( const Domain &dm, std::vector<DblNumVec> &gridpos );

/// @brief Generate a LGL mesh from a domain.
void LGLMesh( const Domain &dm, const Index3& numGrid, std::vector<DblNumVec> &gridpos );


// *********************************************************************
// Formatted output stream
// *********************************************************************

// Bool is NOT defined due to ambiguity with Int

inline Int PrintBlock(std::ostream &os, const std::string name){

  os << std::endl<< "*********************************************************************" << std::endl;
  os << name << std::endl;
  os << "*********************************************************************" << std::endl << std::endl;
  return 0;
}

// String
inline Int Print(std::ostream &os, const std::string name) {
  os << std::setiosflags(std::ios::left) << name << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char* name) {
  os << std::setiosflags(std::ios::left) << std::string(name) << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, std::string val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, const char* val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << std::string(val)
    << std::endl;
  return 0;
};


// Real

// one real number

inline Int Print(std::ostream &os, const std::string name, Real val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char* name, Real val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const std::string name, Real val, const std::string unit) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << unit 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name, Real val, const char *unit) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val  << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos) 
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit) 
    << std::endl;
  return 0;
};

// Two real numbers
inline Int Print(std::ostream &os, const std::string name1, Real val1, const std::string unit1,
    const std::string name2, Real val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name1
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val1 << "  "
    << std::setw(LENGTH_VAR_UNIT) << unit1 
    << std::setw(LENGTH_VAR_NAME) << name2
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2 << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << unit2 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Real val1, const char *unit1,
    char *name2, Real val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val1 << "  "
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2 << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
    << std::endl;
  return 0;
};

// Int and Real
inline Int Print(std::ostream &os, const std::string name1, Int val1, const std::string unit1,
    const std::string name2, Real val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name1
    << std::setw(LENGTH_INT_DATA) << val1 << "  "
    << std::setw(LENGTH_VAR_UNIT) << unit1 
    << std::setw(LENGTH_VAR_NAME) << name2
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2 << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << unit2 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Int val1, const char *unit1,
    char *name2, Real val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1 << "  "
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2 << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
    const char *name1, Int val1, 
    const char *name2, Real val2, 
    char *name3, Real val3) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1  << "  "
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2 << "  "
    << std::setw(LENGTH_VAR_NAME) << std::string(name3) 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3 << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, 
    const char *name1, Int val1, 
    const char *name2, Real val2, 
    const char *name3, Real val3,
    const char *name4, Real val4 ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1  << "  "
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2 << "  "
    << std::setw(LENGTH_VAR_NAME) << std::string(name3) 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3 << "  "
    << std::setw(LENGTH_VAR_NAME) << std::string(name4) 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val4 << "  "
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};

// Int

// one integer number
inline Int Print(std::ostream &os, std::string name, Int val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};
inline Int Print(std::ostream &os, std::string name, int64_t val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char *name, Int val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name, int64_t val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const std::string name, Int val, const std::string unit) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val << "  "
    << std::setw(LENGTH_VAR_UNIT) << unit 
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char* name, Int val, const std::string unit) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setw(LENGTH_VAR_DATA) << val << "  "
    << std::setw(LENGTH_VAR_UNIT) << unit 
    << std::endl;
  return 0;
};



// two integer numbers
inline Int Print(std::ostream &os, const std::string name1, Int val1, const std::string unit1,
    const std::string name2, Int val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name1
    << std::setw(LENGTH_VAR_DATA) << val1 << "  "
    << std::setw(LENGTH_VAR_UNIT) << unit1 
    << std::setw(LENGTH_VAR_NAME) << name2
    << std::setw(LENGTH_VAR_DATA) << val2 << "  "
    << std::setw(LENGTH_VAR_UNIT) << unit2 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Int val1, const char *unit1,
    char *name2, Int val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_VAR_DATA) << val1 << "  "
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << val2 << "  "
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
    << std::endl;
  return 0;
};

// Bool

// one boolean number
inline Int Print(std::ostream &os, const std::string name, bool val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name;
  if( val == true )
    os << std::setw(LENGTH_VAR_NAME) << "true" << std::endl;
  else
    os << std::setw(LENGTH_VAR_NAME) << "false" << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char* name, bool val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name);
  if( val == true )
    os << std::setw(LENGTH_VAR_NAME) << "true" << std::endl;
  else
    os << std::setw(LENGTH_VAR_NAME) << "false" << std::endl;
  return 0;
};


// Index 3 and Point 3
inline Int Print(std::ostream &os, 
    const char *name1, Index3 val ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_VAR_DATA) << val[0] << "  " 
    << std::setw(LENGTH_VAR_DATA) << val[1] << "  " 
    << std::setw(LENGTH_VAR_DATA) << val[2] << "  "
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
    const char *name1, Point3 val ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[0] << "  " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[1] << "  " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[2] << "  " 
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
    const char *name1, Int val1,
    const char *name2, Point3 val ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[0] << "  " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[1] << "  " 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[2] << "  " 
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


// *********************************************************************
// Overload << and >> operators for basic data types
// *********************************************************************

// std::vector
template <class F> inline std::ostream& operator<<( std::ostream& os, const std::vector<F>& vec)
{
  os<<vec.size()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<vec.size(); i++)     
    os<<" "<<vec[i];
  os<<std::endl;
  return os;
}

// NumVec
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumVec<F>& vec)
{
  os<<vec.m()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<vec.m(); i++)     
    os<<" "<<vec(i);
  os<<std::endl;
  return os;
}

//template <class F> inline std::istream& operator>>( std::istream& is, NumVec<F>& vec)
//{
//    Int m;  is>>m;  vec.resize(m);
//    for(Int i=0; i<vec.m(); i++)     
//        is >> vec(i);
//    return is;
//}

// NumMat
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumMat<F>& mat)
{
  os<<mat.m()<<" "<<mat.n()<<std::endl;
  //os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<mat.m(); i++) {
    for(Int j=0; j<mat.n(); j++)
      os<<" "<<mat(i,j);
    os<<std::endl;
  }
  return os;
}

// NumTns
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumTns<F>& tns)
{
  os<<tns.m()<<" "<<tns.n()<<" "<<tns.p()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<tns.m(); i++) {
    for(Int j=0; j<tns.n(); j++) {
      for(Int k=0; k<tns.p(); k++) {
        os<<" "<<tns(i,j,k);
      }
      os<<std::endl;
    }
    os<<std::endl;
  }
  return os;
}

// *********************************************************************
// serialize/deserialize for basic types
// More specific serialize/deserialize will be defined in individual
// class files
// *********************************************************************

//bool
inline Int serialize(const bool& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(bool));
  return 0;
}

inline Int deserialize(bool& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(bool));
  return 0;
}

//char
inline Int serialize(const char& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(char));
  return 0;
}

inline Int deserialize(char& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(char));
  return 0;
}

inline Int combine(char& val, char& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 1;
}

//-------------------
//Int
inline Int serialize(const Int& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(Int));
  return 0;
}

inline Int deserialize(Int& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(Int));
  return 0;
}

inline Int combine(Int& val, Int& ext)
{
  val += ext;
  return 0;
}

//-------------------
//Real
inline Int serialize(const Real& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(Real));
  return 0;
}

inline Int deserialize(Real& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(Real));
  return 0;
}

inline Int combine(Real& val, Real& ext)
{
  val += ext;
  return 0;
}

//-------------------
//Complex
inline Int serialize(const Complex& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(Complex));
  return 0;
}

inline Int deserialize(Complex& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(Complex));
  return 0;
}

inline Int combine(Complex& val, Complex& ext)
{
  val += ext;
  return 0;
}

//-------------------
//Index2  
inline Int serialize(const Index2& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&(val[0]), 2*sizeof(Int));
  return 0;
}

inline Int deserialize(Index2& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&(val[0]), 2*sizeof(Int));
  return 0;
}

inline Int combine(Index2& val, Index2& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

// -------------------
// Point2  
inline Int serialize(const Point2& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&(val[0]), 2*sizeof(Real));
  return 0;
}

inline Int deserialize(Point2& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&(val[0]), 2*sizeof(Real));
  return 0;
}

inline Int combine(Point2& val, Point2& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//Index3
inline Int serialize(const Index3& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&(val[0]), 3*sizeof(Int));
  return 0;
}

inline Int deserialize(Index3& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&(val[0]), 3*sizeof(Int));
  return 0;
}

inline Int combine(Index3& val, Index3& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 1;
}

//-------------------
//Point3
inline Int serialize(const Point3& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&(val[0]), 3*sizeof(Real));
  return 0;
}

inline Int deserialize(Point3& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&(val[0]), 3*sizeof(Real));
  return 0;
}

inline Int combine(Point3& val, Point3& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::vector
template<class T>
Int serialize(const std::vector<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int sz = val.size();
  os.write((char*)&sz, sizeof(Int));
  for(Int k=0; k<sz; k++)
    serialize(val[k], os, mask);
  return 0;
}

template<class T>
Int deserialize(std::vector<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int sz;
  is.read((char*)&sz, sizeof(Int));
  val.resize(sz);
  for(Int k=0; k<sz; k++)
    deserialize(val[k], is, mask);
  return 0;
}

template<class T>
Int combine(std::vector<T>& val, std::vector<T>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::set
template<class T>
Int serialize(const std::set<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int sz = val.size();
  os.write((char*)&sz, sizeof(Int));
  for(typename std::set<T>::const_iterator mi=val.begin(); mi!=val.end(); mi++) 
    serialize((*mi), os, mask);
  return 0;
}

template<class T>
Int deserialize(std::set<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  val.clear();
  Int sz;
  is.read((char*)&sz, sizeof(Int));
  for(Int k=0; k<sz; k++) {
    T t; deserialize(t, is, mask);
    val.insert(t);
  }
  return 0;
}

template<class T>
Int combine(std::set<T>& val, std::set<T>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::map
template<class T, class S>
Int serialize(const std::map<T,S>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int sz = val.size();
  os.write((char*)&sz, sizeof(Int));
  for(typename std::map<T,S>::const_iterator mi=val.begin(); mi!=val.end(); mi++) {
    serialize((*mi).first, os, mask);
    serialize((*mi).second, os, mask);
  }
  return 0;
}

template<class T, class S>
Int deserialize(std::map<T,S>& val, std::istream& is, const std::vector<Int>& mask)
{
  val.clear();
  Int sz;
  is.read((char*)&sz, sizeof(Int));
  for(Int k=0; k<sz; k++) {
    T t;    deserialize(t, is, mask);
    S s;    deserialize(s, is, mask);
    val[t] = s;
  }
  return 0;
}

template<class T, class S>
Int combine(std::map<T,S>& val, std::map<T,S>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::pair
template<class T, class S>
Int serialize(const std::pair<T,S>& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.first, os, mask);
  serialize(val.second, os, mask);
  return 0;
}

template<class T, class S>
Int deserialize(std::pair<T,S>& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.first, is, mask);
  deserialize(val.second, is, mask);
  return 0;
}

template<class T, class S>
Int combine(std::pair<T,S>& val, std::pair<T,S>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

/*
//-------------------
//BolNumVec
inline Int serialize(const BolNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
Int m = val.m();
os.write((char*)&m, sizeof(Int));
os.write((char*)(val.Data()), m*sizeof(bool));
return 0;
}

inline Int deserialize(BolNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
Int m;
is.read((char*)&m, sizeof(Int));
val.Resize(m);
is.read((char*)(val.Data()), m*sizeof(bool));
return 0;
}

//-------------------
//BolNumMat
inline Int serialize(const BolNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
Int m = val.m();
Int n = val.n();
os.write((char*)&m, sizeof(Int));
os.write((char*)&n, sizeof(Int));
os.write((char*)(val.Data()), m*n*sizeof(bool));
return 0;
}

inline Int deserialize(BolNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
Int m;
Int n;
is.read((char*)&m, sizeof(Int));
is.read((char*)&n, sizeof(Int));
val.Resize(m,n);
is.read((char*)(val.Data()), m*n*sizeof(bool));
return 0;
}

//-------------------
//BolNumTns
inline Int serialize(const BolNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
Int m = val.m();  Int n = val.n();  Int p = val.p();
os.write((char*)&m, sizeof(Int));
os.write((char*)&n, sizeof(Int));
os.write((char*)&p, sizeof(Int));
os.write((char*)(val.Data()), m*n*p*sizeof(bool));
return 0;
}

inline Int deserialize(BolNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
Int m,n,p;
is.read((char*)&m, sizeof(Int));
is.read((char*)&n, sizeof(Int));
is.read((char*)&p, sizeof(Int));
val.Resize(m,n,p);
is.read((char*)(val.Data()), m*n*p*sizeof(bool));
return 0;
}
 */

//-------------------
//IntNumVec
inline Int serialize(const IntNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(Int));
  return 0;
}

inline Int deserialize(IntNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(Int));
  return 0;
}

inline Int combine(IntNumVec& val, IntNumVec& ext)
{
  //val.resize(ext.m());
  assert(val.m()==ext.m());
  for(Int i=0; i<val.m(); i++)    val(i) += ext(i);
  return 0;
}


//-------------------
//IntNumMat
inline Int serialize(const IntNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(Int));
  return 0;
}

inline Int deserialize(IntNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(Int));
  return 0;
}

inline Int combine(IntNumMat& val, IntNumMat& ext)
{
  //val.resize(ext.m(),ext.n());
  assert(val.m()==ext.m() && val.n()==ext.n());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      val(i,j) += ext(i,j);
  return 0;
}

//-------------------
//IntNumTns
inline Int serialize(const IntNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(Int));
  return 0;
}

inline Int deserialize(IntNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(Int));
  return 0;
}

inline Int combine(IntNumTns& val, IntNumTns& ext)
{
  //val.resize(ext.m(),ext.n(),ext.p());
  assert(val.m()==ext.m() && val.n()==ext.n() && val.p()==ext.p());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      for(Int k=0; k<val.p(); k++)
        val(i,j,k) += ext(i,j,k);
  return 0;
}

//-------------------
//DblNumVec
inline Int serialize(const DblNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(Real));
  return 0;
}

inline Int deserialize(DblNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(Real));
  return 0;
}

inline Int combine(DblNumVec& val, DblNumVec& ext)
{
  //val.resize(ext.m());
  assert(val.m()==ext.m());
  for(Int i=0; i<val.m(); i++)    val(i) += ext(i);
  return 0;
}

//-------------------
//DblNumMat
inline Int serialize(const DblNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(Real));
  return 0;
}

inline Int deserialize(DblNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(Real));
  return 0;
}

inline Int combine(DblNumMat& val, DblNumMat& ext)
{
  //val.resize(ext.m(),ext.n());
  assert(val.m()==ext.m() && val.n()==ext.n());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      val(i,j) += ext(i,j);
  return 0;
}


//-------------------
//DblNumTns
inline Int serialize(const DblNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(Real));
  return 0;
}

inline Int deserialize(DblNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(Real));
  return 0;
}

inline Int combine(DblNumTns& val, DblNumTns& ext)
{
  //val.resize(ext.m(),ext.n(),ext.p());
  assert(val.m()==ext.m() && val.n()==ext.n() && val.p()==ext.p());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      for(Int k=0; k<val.p(); k++)
        val(i,j,k) += ext(i,j,k);
  return 0;
}

//-------------------
//CpxNumVec
inline Int serialize(const CpxNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(Complex));
  return 0;
}

inline Int deserialize(CpxNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(Complex));
  return 0;
}

inline Int combine(CpxNumVec& val, CpxNumVec& ext)
{
  //val.resize(ext.m());
  assert(val.m()==ext.m());
  for(Int i=0; i<val.m(); i++)    val(i) += ext(i);
  return 0;
}

//-------------------
//CpxNumMat
inline Int serialize(const CpxNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(Complex));
  return 0;
}

inline Int deserialize(CpxNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(Complex));
  return 0;
}

inline Int combine(CpxNumMat& val, CpxNumMat& ext)
{
  //val.resize(ext.m(),ext.n());
  assert(val.m()==ext.m() && val.n()==ext.n());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      val(i,j) += ext(i,j);
  return 0;
}

//-------------------
//CpxNumTns
inline Int serialize(const CpxNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(Complex));
  return 0;
}

inline Int deserialize(CpxNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(Complex));
  return 0;
}

inline Int combine(CpxNumTns& val, CpxNumTns& ext)
{
  //val.resize(ext.m(),ext.n(),ext.p());
  assert(val.m()==ext.m() && val.n()==ext.n() && val.p()==ext.p());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      for(Int k=0; k<val.p(); k++)
        val(i,j,k) += ext(i,j,k);
  return 0;
}

//-------------------
//NumVec
template<class T>
Int inline serialize(const NumVec<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  for(Int i=0; i<m; i++)
    serialize(val(i), os, mask);
  return 0;
}

template<class T>
Int inline deserialize(NumVec<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  for(Int i=0; i<m; i++)
    deserialize(val(i), is, mask);
  return 0;
}

template<class T>
Int inline combine(NumVec<T>& val, NumVec<T>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//NumMat
template<class T>
Int inline serialize(const NumMat<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  for(Int j=0; j<n; j++)
    for(Int i=0; i<m; i++)
      serialize(val(i,j), os, mask);
  return 0;
}
template<class T>
Int inline deserialize(NumMat<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  for(Int j=0; j<n; j++)
    for(Int i=0; i<m; i++)
      deserialize(val(i,j), is, mask);
  return 0;
}

template<class T>
Int inline combine(NumMat<T>& val, NumMat<T>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}


//-------------------
//NumTns
template<class T>
Int inline serialize(const NumTns<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  for(Int k=0; k<p; k++)
    for(Int j=0; j<n; j++)
      for(Int i=0; i<m; i++)
        serialize(val(i,j,k), os, mask);
  return 0;
}

template<class T>
Int inline deserialize(NumTns<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  Int p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  for(Int k=0; k<p; k++)
    for(Int j=0; j<n; j++)
      for(Int i=0; i<m; i++)
        deserialize(val(i,j,k), is, mask);
  return 0;
}

template<class T>
Int inline combine(NumTns<T>& val, NumTns<T>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

Int inline serialize(const Domain& dm, std::ostream& os, const std::vector<Int>& mask){
  serialize( dm.length, os, mask );
  serialize( dm.posStart, os, mask );
  serialize( dm.numGrid, os, mask );
  // Do not serialize the communicatior
  return 0;
}

Int inline deserialize(Domain& dm, std::istream& is, const std::vector<Int>& mask){
  deserialize( dm.length, is, mask );
  deserialize( dm.posStart, is, mask );
  deserialize( dm.numGrid, is, mask );
  // Do not deserialize the communicatior 
  return 0;
}

Int inline combine(Domain& dm1, Domain& dm2){
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}

//-------------------
//DistSparseMatrix
template<class T>
Int inline serialize(const DistSparseMatrix<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize( val.size,        os, mask );
  serialize( val.nnz,         os, mask );
  serialize( val.nnzLocal,    os, mask );
  serialize( val.firstCol,    os, mask );
  serialize( val.colptrLocal, os, mask );
  serialize( val.rowindLocal, os, mask );
  serialize( val.nzvalLocal,  os, mask );
  // No need to serialize the communicator
  return 0;
}

template<class T>
Int inline deserialize(DistSparseMatrix<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize( val.size,        is, mask );
  deserialize( val.nnz,         is, mask );
  deserialize( val.nnzLocal,    is, mask );
  deserialize( val.firstCol,    is, mask );
  deserialize( val.colptrLocal, is, mask );
  deserialize( val.rowindLocal, is, mask );
  deserialize( val.nzvalLocal,  is, mask );
  // No need to deserialize the communicator
  return 0;
}

template<class T>
Int inline combine(DistSparseMatrix<T>& val, DistSparseMatrix<T>& ext)
{
  ErrorHandling( "Combine operation not implemented." );
  return 0;
}


// *********************************************************************
// Parallel IO functions
// *********************************************************************

Int SeparateRead(std::string name, std::istringstream& is);

Int SeparateRead(std::string name, std::istringstream& is, Int outputIndex);

Int SeparateWrite(std::string name, std::ostringstream& os);

Int SeparateWrite(std::string name, std::ostringstream& os, Int outputIndex);

Int SeparateWriteAscii(std::string name, std::ostringstream& os);

Int SharedRead(std::string name, std::istringstream& is);

Int SharedWrite(std::string name, std::ostringstream& os);


// *********************************************************************
// Random numbers
// *********************************************************************
inline void SetRandomSeed(long int seed){
  srand48(seed);
}

inline Real UniformRandom(){
  return (Real)drand48();
}

inline void UniformRandom( NumVec<Real>& vec )
{
  for(Int i=0; i<vec.m(); i++)
    vec(i) = UniformRandom();
}

inline void UniformRandom( NumVec<Complex>& vec )
{
  for(Int i=0; i<vec.m(); i++)
    vec(i) = Complex(UniformRandom(), UniformRandom());
}

inline void UniformRandom( NumMat<Real>& M )
{
  Real *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
    *(ptr++) = UniformRandom(); 
}

inline void UniformRandom( NumMat<Complex>& M )
{
  Complex *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
    *(ptr++) = Complex(UniformRandom(), UniformRandom()); 
}


inline void UniformRandom( NumTns<Real>& T )
{
  Real *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
    *(ptr++) = UniformRandom(); 
}

inline void UniformRandom( NumTns<Complex>& T )
{
  Complex *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
    *(ptr++) = Complex(UniformRandom(), UniformRandom()); 
}

inline Real GaussianRandom(){
  // Box-Muller method for generating a random Gaussian number
  Real a = UniformRandom(), b = UniformRandom();
  return std::sqrt(-2.0*std::log(a))*std::cos(2.0*PI*b);
}

inline void GaussianRandom( NumVec<Real>& vec )
{
  for(Int i=0; i<vec.m(); i++)
    vec(i) = GaussianRandom();
}

inline void GaussianRandom( NumVec<Complex>& vec )
{
  for(Int i=0; i<vec.m(); i++)
    vec(i) = Complex(GaussianRandom(), GaussianRandom());
}

inline void GaussianRandom( NumMat<Real>& M )
{
  Real *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
    *(ptr++) = GaussianRandom(); 
}

inline void GaussianRandom( NumMat<Complex>& M )
{
  Complex *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
    *(ptr++) = Complex(GaussianRandom(), GaussianRandom()); 
}


inline void GaussianRandom( NumTns<Real>& T )
{
  Real *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
    *(ptr++) = GaussianRandom(); 
}

inline void GaussianRandom( NumTns<Complex>& T )
{
  Complex *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
    *(ptr++) = Complex(GaussianRandom(), GaussianRandom()); 
}


// *********************************************************************
// Timing
// *********************************************************************
inline void GetTime(Real&  t){
  t = MPI_Wtime();
}

// *********************************************************************
// Comparator
// *********************************************************************

// Real
inline bool PairLtComparator( const std::pair<Real, Int>& l, 
    const std::pair<Real, Int>& r ){
  return l.first < r.first;
}

inline bool PairGtComparator( const std::pair<Real, Int>& l, 
    const std::pair<Real, Int>& r ){
  return l.first > r.first;
}

// For sorting with indices
// Example usage:
//   std::sort(val.begin(), val.end(), IndexComp<std::vector<int>&>(indices));
template<class T> 
struct IndexComp {
private: 
  const T indices_;
public:
  IndexComp (const T indices) : indices_(indices) {}
  bool operator()(const size_t a, const size_t b) const
  { return indices_[a] < indices_[b]; }
};


// *********************************************************************
// Sparse Matrix
// *********************************************************************

// TODO Complex format
void ReadSparseMatrix ( const char* filename, SparseMatrix<Real>& spmat );

void ReadDistSparseMatrix( const char* filename, DistSparseMatrix<Real>& pspmat, MPI_Comm comm );

void ReadDistSparseMatrixFormatted( const char* filename, DistSparseMatrix<Real>& pspmat, MPI_Comm comm );

void WriteDistSparseMatrixFormatted( const char* filename, const DistSparseMatrix<Real>& pspmat);

template <class F1, class F2> 
void
CopyPattern    ( const SparseMatrix<F1>& A, SparseMatrix<F2>& B )
{
  B.size        = A.size;
  B.nnz         = A.nnz;
  B.colptr      = A.colptr;
  B.rowind      = A.rowind;
  B.nzval.Resize( A.nnz );
  return ;
}        // -----  end of template function CopyPattern  ----- 


// Functions for DistSparseMatrix

template <class F1, class F2> 
void
CopyPattern    ( const DistSparseMatrix<F1>& A, DistSparseMatrix<F2>& B )
{
  B.size        = A.size;
  B.nnz         = A.nnz;
  B.nnzLocal    = A.nnzLocal;
  B.firstCol    = A.firstCol;
  B.colptrLocal = A.colptrLocal;
  B.rowindLocal = A.rowindLocal;
  B.nzvalLocal.Resize( A.nnzLocal );
  B.comm        = A.comm;
  return ;
}        // -----  end of template function CopyPattern  ----- 


#if 0
void AlltoallForward( DblNumMat& A, DblNumMat& B, MPI_Comm comm );
void AlltoallBackward( DblNumMat& A, DblNumMat& B, MPI_Comm comm );
void AlltoallForward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm );
void AlltoallBackward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm );
#endif

#ifdef _COMPLEX_
#ifdef GPU
void GPU_AlltoallBackward( cuCpxNumMat& A, cuCpxNumMat& B, MPI_Comm comm );
void GPU_AlltoallForward ( cuCpxNumMat& A, cuCpxNumMat& B, MPI_Comm comm );
#endif
#else
#ifdef GPU
void GPU_AlltoallBackward( cuDblNumMat& A, cuDblNumMat& B, MPI_Comm comm );
void GPU_AlltoallForward ( cuDblNumMat& A, cuDblNumMat& B, MPI_Comm comm );
#endif
#endif
// ~~**~~
// ------------------------------------------------------------------------------
// Simple distributor class: parallel work load distributor.
// Given an array of size sz on which operations
// have to be distributed between num_procs procesors,
// this class enables us to distribute the work as equally as
// possible (according to a round robin fashion).
class simple_distributor
{
private:
  int sz;
  int num_procs;

public:
  std::vector <int> start_ind;
  std::vector <int> end_ind;
  std::vector <int> size_list;
  int current_proc_start;
  int current_proc_end;
  int current_proc_size;

  simple_distributor(int inp_sz, int inp_nprocs, int current_proc_rank)
  {
    // Set input values
    sz = inp_sz;
    num_procs = inp_nprocs;

    // Assign arrays
    start_ind.resize(num_procs);
    end_ind.resize(num_procs);
    size_list.resize(num_procs);

    // Distribute the sizes here
    int iter;
    int eq = sz / num_procs, rem = sz % num_procs;

    // Assign equal portion to everyone
    for(iter = 0; iter < num_procs; iter++)
      size_list[iter] = eq;

    // Distribute the rest
    for(iter = 0; iter < rem; iter++)
      size_list[iter]++;

    // Now figure out the start and end indices for all
    start_ind[0] = 0;
    end_ind[0] = size_list[0] - 1;
    for(iter = 1; iter < num_procs ; iter++)    
    {
      start_ind[iter] = end_ind[iter - 1] + 1;
      end_ind[iter] = start_ind[iter] + size_list[iter] - 1;
    }

    // Finally setup the values for the current processor
    current_proc_start = start_ind[current_proc_rank];
    current_proc_end = end_ind[current_proc_rank];
    current_proc_size = size_list[current_proc_rank];
  }

};


// serialize/deserialize the pseudopot

Int serialize(const PseudoPot& val, std::ostream& os, const std::vector<Int>& mask);

Int deserialize(PseudoPot& val, std::istream& is, const std::vector<Int>& mask);

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin);

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin, NumVec<Real>& amin);

void pdist2(NumMat<Real>& A, NumMat<Real>& B, NumMat<Real>& D);

void unique(NumVec<Int>& Index);

void KMEAN(Int n, NumVec<Real>& weight, Int& rk, Real KmeansTolerance, 
    Int KmeansMaxIter, Real DFTolerance, const Domain &dm, Int* piv);

void spline(int n, double *x, double *y, double yp_left, double yp_right,
            int bcnat_left, int bcnat_right, double *y2);
void splint (int n, double *xa, double *ya, double *y2a, double x, double *y);
void splintd (int n, double *xa, double *ya, double *y2a,
              double x, double *y, double *dy);
std::string find_start_element(std::string name, std::ifstream &upfin);
void find_end_element(std::string name, std::ifstream & upfin);
void seek_str(std::string tag, std::ifstream &upfin);
std::string get_attr(std::string buf, std::string attr);
void skipln(std::ifstream &upfin);

// spline along the radial direction
void splinerad( std::vector<double> & r, std::vector<double> & vloc, std::vector<double> & out_r, std::vector<double> & out_vloc , int even);
} // namespace scales
#endif // _UTILITY_HPP_
