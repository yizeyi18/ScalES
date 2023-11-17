//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin

/// @file sparse_matrix_impl.hpp
/// @brief Implementation of sparse matrices.
/// @date 2012-11-28
#ifndef _SPARSE_MATRIX_IMPL_HPP_
#define _SPARSE_MATRIX_IMPL_HPP_

#include "sparse_matrix_decl.h"
#include "mpi_interf.h"

namespace  scales{

extern Int SharedRead(std::string name, std::istringstream& is);        //实现见utility.cpp

//---------------------------------------------------------
template<typename F>
  void ReadSparseMatrix ( const char* filename, SparseMatrix<F>& spmat )
  {

    std::istringstream iss;
    Int dummy;                                                          //干什么的？疑似只是和Formatted版保持一致。可以删？
    SharedRead( std::string(filename), iss );
    deserialize( spmat.size, iss, NO_MASK );                            //在utility.h声明。为什么这儿能用？？
    deserialize( spmat.dummy, iss, NO_MASK );
    deserialize( spmat.nnz,  iss, NO_MASK );
    deserialize( spmat.colptr, iss, NO_MASK );
    deserialize( spmat.rowind, iss, NO_MASK );
    deserialize( spmat.nzval, iss, NO_MASK );


    return ;
  }        // -----  end of function ReadSparseMatrix  ----- 


template <class F> void
  ReadSparseMatrixFormatted    ( const char* filename, SparseMatrix<F>& spmat )
  {
    std::ifstream fin(filename);
    Int dummy;
    fin >> spmat.size >> dummy >> spmat.nnz;

    spmat.colptr.Resize( spmat.size+1 );//+1是"FORTRAN-convention"的结果？
    spmat.rowind.Resize( spmat.nnz );   //回上行：不是，是colptr[-1]=spmat.nnz的
    spmat.nzval.Resize ( spmat.nnz );   //结果。这个冗余colptr可以少判断一次到没到最后一列。

    for( Int i = 0; i < spmat.size + 1; i++ ){
      fin >> spmat.colptr(i);
    }

    for( Int i = 0; i < spmat.nnz; i++ ){
      fin >> spmat.rowind(i);
    }

    for( Int i = 0; i < spmat.nnz; i++ ){
      fin >> spmat.nzval(i);
    }

    fin.close();

    return ;
  }        // -----  end of function ReadSparseMatrixFormatted  ----- 

//---------------------------------------------------------
template<typename F>
  void ReadDistSparseMatrix ( const char* filename, DistSparseMatrix<F>& pspmat, MPI_Comm comm )
  {
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    std::ifstream fin;


    // Read basic information
    if( mpirank == 0 ){
      fin.open(filename);
      if( !fin.good() ){
        ErrorHandling( "File cannot be openeded!" );
      }
      Int dummy;
      fin.read((char*)&pspmat.size, sizeof(Int));
      fin.read((char*)&dummy, sizeof(Int));
      fin.read((char*)&pspmat.nnz,  sizeof(Int));
    }//if mpirank == 0

    pspmat.comm = comm;

    MPI_Bcast(&pspmat.size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&pspmat.nnz,  1, MPI_INT, 0, comm);

    // Read colptr

    IntNumVec  colptr(pspmat.size+1);
    if( mpirank == 0 ){
      Int tmp;
      fin.read((char*)&tmp, sizeof(Int));  
      if( tmp != pspmat.size+1 ){//DistSparseMatrix二进制文件会单存一个colptr长度而非默认为size+1
        ErrorHandling( "colptr is not of the right size." );
      }
      fin.read((char*)colptr.Data(), sizeof(Int)*tmp);//由rank 0独立读这个O(N)的量
    }

    MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);
    //    std::cout << "Proc " << mpirank << " outputs colptr[end]" << colptr[pspmat.size] << endl;


    // Compute the number of columns on each processor
    IntNumVec numColLocalVec(mpisize);//长为mpisize的numvec，存储各个MPI进程本地列数，由每个MPI进程独立求算
    Int numColLocal, numColFirst;//本进程本地列数与非最末进程列数
    numColFirst = pspmat.size / mpisize;//不是本地首列，非最末进程都是这么多列，不知道为什么起这个名字
    SetValue( numColLocalVec, numColFirst );
    numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry最后一个进程多干点
    numColLocal = numColLocalVec[mpirank];//此时确定下来本地列数
    pspmat.colptrLocal.Resize( numColLocal + 1 );
    for( Int i = 0; i < numColLocal + 1; i++ ){//此循环中用mpirank*numColFirst代替真正的本进程首列。
      pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
    }//如果要改变最后一个进程多干的情况，就把mpirank*numColFirst换成sum(numColLocalVec[0:mpirank])。
    //备忘:rowindLocal[colptrLocal[n]]~rowindLocal[colptrLocal[n+1]]是属于本地第n列的非0元。


    // Calculate nnz_loc on each processor
    pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];//末减初，安全性可能不大好；应用colptrLocal.m()代替numColLocal
    pspmat.rowindLocal.Resize( pspmat.nnzLocal );
    pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );


    // Read and distribute the row indices
    if( mpirank == 0 ){
      Int tmp;
      fin.read((char*)&tmp, sizeof(Int));  
      if( tmp != pspmat.nnz ){
        std::ostringstream msg;
        msg 
          << "The number of nonzeros and row indices do not match." << std::endl
          << "nnz = " << pspmat.nnz << std::endl
          << "size of row indices = " << tmp << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      IntNumVec buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){//极限条件下是O(N^2)的复杂度，但没并行
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];//ip*numColFirst代替真正的rank i首列
        buf.Resize(numRead);
        fin.read( (char*)buf.Data(), numRead*sizeof(Int) );//每个rank的读都由rank 0完成，分别发送
        if( ip > 0 ){
          MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
          MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);//小飞棍来喽！
        }
        else{
          pspmat.rowindLocal = buf;
        }
      }
    }//if mpirank == 0
    else{// mpirank != 0
      Int numRead;
      MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
      if( numRead != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in row indices do not match." << std::endl
          << "numRead  = " << numRead << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }

      pspmat.rowindLocal.Resize( numRead );
      MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
    }

    //    std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
    //        << pspmat.rowindLocal.m() << endl;


    // Read and distribute the nonzero values
    if( mpirank == 0 ){//同上，没有并行
      Int tmp;
      fin.read((char*)&tmp, sizeof(Int));  
      if( tmp != pspmat.nnz ){
        std::ostringstream msg;
        msg 
          << "The number of nonzeros in values do not match." << std::endl
          << "nnz = " << pspmat.nnz << std::endl
          << "size of values = " << tmp << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      NumVec<F> buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){//由rank 0读各个rank的非0并发给对应rank
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];//同上，ip*numColFirst代替真正的rank i首列
        buf.Resize(numRead);
        fin.read( (char*)buf.Data(), numRead*sizeof(F) );
        if( ip > 0 ){
          std::stringstream sstm;//与上不同，没有发送buf.Data()+numRead而是用serialize
          serialize( buf, sstm, NO_MASK );//将buf存入一stringstream后发送。
          mpi::Send( sstm, ip, 0, 1, comm );//serialize是干什么的？似乎只是把numvec转成stream？
        }
        else{
          pspmat.nzvalLocal = buf;
        }
      }//for
    }//if mpirank == 0
    else{//mpirank != 0
      std::stringstream sstm;
      mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
      deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
      if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in values do not match." << std::endl
          << "numRead  = " << pspmat.nzvalLocal.m() << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

    // Close the file
    if( mpirank == 0 ){
      fin.close();
    }



    MPI_Barrier( comm );


    return ;
  }        // -----  end of function ReadDistSparseMatrix  ----- 



template<typename F>
  void ReadDistSparseMatrixFormatted ( const char* filename, DistSparseMatrix<F>& pspmat, MPI_Comm comm )
  {
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    std::ifstream fin;

    // Read basic information
    if( mpirank == 0 ){
      fin.open(filename);
      if( !fin.good() ){
        ErrorHandling( "File cannot be openeded!" );
      }
      Int dummy;
      fin >> pspmat.size >> dummy;
      fin >> pspmat.nnz;
    }

    pspmat.comm = comm;

    MPI_Bcast(&pspmat.size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&pspmat.nnz,  1, MPI_INT, 0, comm);

    // Read colptr

    IntNumVec  colptr(pspmat.size+1);
    if( mpirank == 0 ){
      Int* ptr = colptr.Data();
      for( Int i = 0; i < pspmat.size+1; i++ )
        fin >> *(ptr++);
    }

    MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);

    // Compute the number of columns on each processor
    IntNumVec numColLocalVec(mpisize);//同上，此变量记录各进程本地列数
    Int numColLocal, numColFirst;//本进程本地列数，非最末进程本地列数
    numColFirst = pspmat.size / mpisize;//除最末进程外各进程本地列数都是这个数！
    SetValue( numColLocalVec, numColFirst );
    numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry    
    numColLocal = numColLocalVec[mpirank];//最末进程独吞size%mpisize

    // The first column follows the 1-based (FORTRAN convention) index.
    pspmat.firstCol = mpirank * numColFirst + 1;

    pspmat.colptrLocal.Resize( numColLocal + 1 );
    for( Int i = 0; i < numColLocal + 1; i++ ){
      pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
    }

    // Calculate nnz_loc on each processor
    pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];//末减初

    pspmat.rowindLocal.Resize( pspmat.nnzLocal );
    pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

    // Read and distribute the row indices
    if( mpirank == 0 ){//同上，rank 0读，再发送给别的进程
      Int tmp;
      IntNumVec buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];
        buf.Resize(numRead);
        Int *ptr = buf.Data();
        for( Int i = 0; i < numRead; i++ ){
          fin >> *(ptr++);
        }
        if( ip > 0 ){
          MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
          MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);
        }
        else{
          pspmat.rowindLocal = buf;
        }
      }//for
    }//if mpirank == 0
    else{//mpirank ！= 0
      Int numRead;
      MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
      if( numRead != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in row indices do not match." << std::endl
          << "numRead  = " << numRead << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }

      pspmat.rowindLocal.Resize( numRead );
      MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
    }

#if ( _DEBUGlevel_ >= 2 )
    std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
      << pspmat.rowindLocal.m() << endl;
#endif


    // Read and distribute the nonzero values
    if( mpirank == 0 ){
      Int tmp;
      NumVec<F> buf;
      Int numRead;
      for( Int ip = 0; ip < mpisize; ip++ ){
        numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
          colptr[ip*numColFirst];
        buf.Resize(numRead);
        F *ptr = buf.Data();
        for( Int i = 0; i < numRead; i++ ){
          fin >> *(ptr++);
        }
        if( ip > 0 ){
          std::stringstream sstm;
          serialize( buf, sstm, NO_MASK );//所以为什么rowIndex和nonZero还不一样？
          mpi::Send( sstm, ip, 0, 1, comm );//写个sizeof(T)就这么不可接受？
        }
        else{
          pspmat.nzvalLocal = buf;
        }
      }
    }
    else{
      std::stringstream sstm;
      mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
      deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
      if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
        std::ostringstream msg;
        msg << "The number of columns in values do not match." << std::endl
          << "numRead  = " << pspmat.nzvalLocal.m() << std::endl
          << "nnzLocal = " << pspmat.nnzLocal << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

    // Close the file
    if( mpirank == 0 ){
      fin.close();
    }

    MPI_Barrier( comm );


    return ;
  }        // -----  end of function ReadDistSparseMatrixFormatted  ----- 


template<typename F>
  void WriteDistSparseMatrixFormatted ( 
      const char* filename, 
      DistSparseMatrix<F>& pspmat    )//神奇的缩进
  {
    // Get the processor information within the current communicator
    MPI_Comm comm = pspmat.comm;
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);

    MPI_Status mpistat;
    std::ofstream ofs;

    // Write basic information
    if( mpirank == 0 ){
      ofs.open(filename, std::ios_base::out);
      if( !ofs.good() ){
        ErrorHandling( "File cannot be opened!" );
      }
      ofs << std::setiosflags(std::ios::left) 
        << std::setw(LENGTH_VAR_DATA) << pspmat.size//LENGTH_VAR_DATA在environment.h定义，控制多处小数输出精度
        << std::setw(LENGTH_VAR_DATA) << pspmat.size
        << std::setw(LENGTH_VAR_DATA) << pspmat.nnz << std::endl;
      ofs.close();
    }

    // Write colptr information, one processor after another
    IntNumVec colptrSizeLocal(mpisize);//存储本进程本地nonzero数量。为什么要叫colptrsize？
    SetValue( colptrSizeLocal, 0 );
    IntNumVec colptrSize(mpisize);//Allreduce过的colptrSize，存储各进程本地nonzero数量
    SetValue( colptrSize, 0 );
    colptrSizeLocal(mpirank) = pspmat.colptrLocal[pspmat.colptrLocal.Size()-1] - 1;//甚至再算一遍nnzLocal，不理解，有何门道吗？
    mpi::Allreduce( colptrSizeLocal.Data(), colptrSize.Data(),
        mpisize, MPI_SUM, comm );//在mpi_interf中实现，通过重载时指定指针数据类型避开了手动指定数据类型
    IntNumVec colptrStart(mpisize);//各进程的firstcol，Fortran风格
    colptrStart[0] = 1;
    for( Int l = 1; l < mpisize; l++ ){//甚至还要现场算一遍，真想即刻重写掉......
      colptrStart[l] = colptrStart[l-1] + colptrSize[l-1];
    }
    for( Int p = 0; p < mpisize; p++ ){
      if( mpirank == p ){//不做并行！就要线性写入！兼容性压倒一切！
        ofs.open(filename, std::ios_base::out | std::ios_base::app );
        if( !ofs.good() ){
          ErrorHandling( "File cannot be openeded!" );
        }
        IntNumVec& colptrLocal = pspmat.colptrLocal;
        for( Int i = 0; i < colptrLocal.Size() - 1; i++ ){//为什么是<size-1？colptrLocal的最后一个不要了？
          ofs << std::setiosflags(std::ios::left) //——答案是colptrlocal有一个冗余，它的尾巴就是下一个的头
            << colptrLocal[i] + colptrStart[p] - 1 << "  ";//用Fortran风格的代价！-1！
        }
        if( p == mpisize - 1 ){
          ofs << std::setiosflags(std::ios::left) 
            << colptrLocal[colptrLocal.Size()-1] + colptrStart[p] - 1 << std::endl;
        }//最后一个进程可以写尾巴，这也是全局的尾巴
        ofs.close();//写完就关
      }//if mpirank == p

      MPI_Barrier( comm );
    }//for p

    // Write rowind information, one processor after another
    for( Int p = 0; p < mpisize; p++ ){//同上，纯串行，不过没有尾巴问题
      if( mpirank == p ){
        ofs.open(filename, std::ios_base::out | std::ios_base::app );
        if( !ofs.good() ){
          ErrorHandling( "File cannot be openeded!" );
        }
        IntNumVec& rowindLocal = pspmat.rowindLocal;
        for( Int i = 0; i < rowindLocal.Size(); i++ ){
          ofs << std::setiosflags(std::ios::left) 
            << rowindLocal[i] << "  ";
        }
        if( p == mpisize - 1 ){
          ofs << std::endl;
        }
        ofs.close();
      }//if mpirank == p

      MPI_Barrier( comm );
    }//for p

    // Write nzval information, one processor after another
    for( Int p = 0; p < mpisize; p++ ){
      if( mpirank == p ){
        ofs.open(filename, std::ios_base::out | std::ios_base::app );
        if( !ofs.good() ){
          ErrorHandling( "File cannot be openeded!" );
        }
        NumVec<F>& nzvalLocal = pspmat.nzvalLocal;
        for( Int i = 0; i < nzvalLocal.Size(); i++ ){
          ofs << std::setiosflags(std::ios::left) 
            << std::setiosflags(std::ios::scientific)
            << std::setiosflags(std::ios::showpos)
            << std::setprecision(LENGTH_FULL_PREC)
            << nzvalLocal[i] << "  ";
        }
        if( p == mpisize - 1 ){
          ofs << std::endl;
        }
        ofs.close();
      }//if mpirank == p

      MPI_Barrier( comm );
    }//for p

    MPI_Barrier( comm );//与for里的barrier相邻。有何用？
    return ;
  }        // -----  end of function WriteDistSparseMatrixFormatted  ----- 

template<typename F>
  void ParaReadDistSparseMatrix ( //并行！
      const char* filename, 
      DistSparseMatrix<F>& pspmat,
      MPI_Comm comm    )
  {
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    MPI_Datatype type;
    Int lens[3];//Magic Number。为什么是3呢？
    MPI_Aint disps[3];//MPI_Aint用来存各式地址，换句话说，相当于void*
    MPI_Datatype types[3];
    Int err = 0;

    Int filemode = MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN;//按位与是什么鬼？？？
							  //基本上就是MPI_MODE_UNIQUE_OPEN？
							  //不会是给什么平台埋坑的吧
    MPI_File fin;
    MPI_Status status;
    err = MPI_File_open(comm,(char*) filename, filemode, MPI_INFO_NULL,  &fin);
    if (err != MPI_SUCCESS) 
      ErrorHandling( "File cannot be opened!" );

    // Read header
    if( mpirank == 0 ){
      err = MPI_File_read_at(fin, 0,(char*)&pspmat.size, 1, MPI_INT, &status);
      err = MPI_File_read_at(fin, sizeof(Int),(char*)&pspmat.nnz, 1, MPI_INT, &status);
    }


    /* define a struct that describes all our data */
    lens[0] = 1;
    lens[1] = 1;
    MPI_Get_address(&pspmat.size, &disps[0]);
    MPI_Get_address(&pspmat.nnz, &disps[1]);
    types[0] = MPI_INT;
    types[1] = MPI_INT;
    MPI_Type_create_struct(2, lens, disps, types, &type);//向type写入以lens，disps，types描述的MPI结构体类型
    MPI_Type_commit(&type);//MPI类型创建完后显式调用commit完成类型向MPI声明，可以用于通信

    /* broadcast the header data to everyone */
    MPI_Bcast(MPI_BOTTOM, 1, type, 0, comm);//于是每个进程就都拿到了rank 0读到的pspmat.size和pspmat.nnz

    MPI_Type_free(&type);//它用来广播pspmat.size和pspmat.nnz的历史使命完成啦！
			 //疑惑：为什么不直接分别Bcast pspmat.size和pspmat.nnz这两个量而要创个结构体？

    // Compute the number of columns on each processor
    IntNumVec numColLocalVec(mpisize);
    Int numColLocal, numColFirst;//同串行read
    numColFirst = pspmat.size / mpisize;
    SetValue( numColLocalVec, numColFirst );
    numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry    
    numColLocal = numColLocalVec[mpirank];
    pspmat.colptrLocal.Resize( numColLocal + 1 );

    //并行读核心科技：计算笨进程读取起点，即下面的Offset
    //Magic Number 2来自前述size与nnz所占空间
    //rank 0少偏移的一个sizeof(Int)来自什么？

    MPI_Offset myColPtrOffset = (2 + ((mpirank==0)?0:1) )*sizeof(Int) + (mpirank*numColFirst)*sizeof(Int);

    Int np1 = 0;//干什么的？似乎是串行时用来核对待读数据数量的，并行中终于意识到不需要它了但也没删
    lens[0] = (mpirank==0)?1:0;
    lens[1] = numColLocal + 1;//+1来自冗余列

    MPI_Get_address(&np1, &disps[0]);
    MPI_Get_address(pspmat.colptrLocal.Data(), &disps[1]);

    MPI_Type_create_hindexed(2, lens, disps, MPI_INT, &type);
    MPI_Type_commit(&type);

    err= MPI_File_read_at_all(fin, myColPtrOffset, MPI_BOTTOM, 1, type, &status);//读列信息

    if (err != MPI_SUCCESS) {
      ErrorHandling( "error reading colptr" );
    }
    MPI_Type_free(&type);

    // Calculate nnz_loc on each processor
    pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];//末减初


    pspmat.rowindLocal.Resize( pspmat.nnzLocal );
    pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

    //read rowIdx
    //跳过头、整个列部分与别的进程的行
    MPI_Offset myRowIdxOffset = (2 + ((mpirank==0)?0:1) )*sizeof(Int) + (pspmat.size+1 + pspmat.colptrLocal[0])*sizeof(Int);

    lens[0] = (mpirank==0)?1:0;
    lens[1] = pspmat.nnzLocal;

    MPI_Get_address(&np1, &disps[0]);
    MPI_Get_address(pspmat.rowindLocal.Data(), &disps[1]);

    MPI_Type_create_hindexed(2, lens, disps, MPI_INT, &type);
    MPI_Type_commit(&type);

    err= MPI_File_read_at_all(fin, myRowIdxOffset, MPI_BOTTOM, 1, type,&status);//读行部分

    if (err != MPI_SUCCESS) {
      ErrorHandling( "error reading rowind/row index" );
    }
    MPI_Type_free(&type);


    //read nzval
    //跳过头、整个列部分、整个行部分与别的进程的非0
    MPI_Offset myNzValOffset = (2 + ((mpirank==0)?0:1) )*sizeof(Int) + (pspmat.size+1 + pspmat.nnz)*sizeof(Int) + pspmat.colptrLocal[0]*sizeof(F);

    lens[0] = (mpirank==0)?1:0;
    lens[1] = pspmat.nnzLocal;

    MPI_Get_address(&np1, &disps[0]);
    MPI_Get_address(pspmat.nzvalLocal.Data(), &disps[1]);

    types[0] = MPI_INT;
    // FIXME Currently only support double format
    // 应该加个if就行？
    // 所以还有什么类型的数据需要读呢
    if( sizeof(F) != sizeof(double) ){
      ErrorHandling("ParaReadDistSparseMatrix only supports format with a size equal to double");
    }

    types[1] = MPI_DOUBLE;

    MPI_Type_create_struct(2, lens, disps, types, &type);
    MPI_Type_commit(&type);

    err = MPI_File_read_at_all(fin, myNzValOffset, MPI_BOTTOM, 1, type,&status);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "error reading nzval" );
    }

    MPI_Type_free(&type);


    //convert to local references
    for( Int i = 1; i < numColLocal + 1; i++ ){
      pspmat.colptrLocal[i] = pspmat.colptrLocal[i] -  pspmat.colptrLocal[0] + 1;
    }
    pspmat.colptrLocal[0]=1;

    MPI_Barrier( comm );

    MPI_File_close(&fin);

    return ;
  }        // -----  end of function ParaReadDistSparseMatrix  ----- 

template<typename F>
  void
  ParaWriteDistSparseMatrix ( 
      const char* filename, 
      DistSparseMatrix<F>& pspmat )
  {
    MPI_Comm  comm = pspmat.comm;
    // Get the processor information within the current communicator
    MPI_Barrier( comm );
    Int mpirank;  MPI_Comm_rank(comm, &mpirank);
    Int mpisize;  MPI_Comm_size(comm, &mpisize);
    MPI_Status mpistat;
    Int err = 0;



    int filemode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_UNIQUE_OPEN;

    MPI_File fout;
    MPI_Status status;



    err = MPI_File_open(comm,(char*) filename, filemode, MPI_INFO_NULL,  &fout);

    if (err != MPI_SUCCESS) {
      ErrorHandling( "File cannot be opened!" );
    }

    // Write header
    if( mpirank == 0 ){
      err = MPI_File_write_at(fout, 0,(char*)&pspmat.size, 1, MPI_INT, &status);
      err = MPI_File_write_at(fout, sizeof(Int),(char*)&pspmat.nnz, 1, MPI_INT, &status);
    }


    // Compute the number of columns on each processor
    Int numColLocal = pspmat.colptrLocal.m()-1;
    Int numColFirst = pspmat.size / mpisize;
    IntNumVec  colptrChunk(numColLocal+1);

    Int prev_nz = 0;
    MPI_Exscan(&pspmat.nnzLocal, &prev_nz, 1, MPI_INT, MPI_SUM, comm);

    for( Int i = 0; i < numColLocal + 1; i++ ){
      colptrChunk[i] = pspmat.colptrLocal[i] + prev_nz;
    }


    MPI_Datatype memtype, filetype;
    MPI_Aint disps[6];
    int blklens[6];
    // FIXME Currently only support double format
    if( sizeof(F) != sizeof(double) ){
      ErrorHandling("ParaReadDistSparseMatrix only supports double format");
    }

    MPI_Datatype types[6] = {MPI_INT,MPI_INT, MPI_INT,MPI_INT, MPI_INT,MPI_DOUBLE};

    /* set block lengths (same for both types) */
    blklens[0] = (mpirank==0)?1:0;
    blklens[1] = numColLocal+1;
    blklens[2] = (mpirank==0)?1:0;
    blklens[3] = pspmat.nnzLocal;
    blklens[4] = (mpirank==0)?1:0;
    blklens[5] = pspmat.nnzLocal;




    //Calculate offsets
    MPI_Offset myColPtrOffset, myRowIdxOffset, myNzValOffset;
    myColPtrOffset = 3*sizeof(Int) + (mpirank*numColFirst)*sizeof(Int);
    myRowIdxOffset = 3*sizeof(Int) + (pspmat.size +1  +  prev_nz)*sizeof(Int);
    myNzValOffset = 4*sizeof(Int) + (pspmat.size +1 +  pspmat.nnz)*sizeof(Int)+ prev_nz*sizeof(F);
    disps[0] = 2*sizeof(Int);
    disps[1] = myColPtrOffset;
    disps[2] = myRowIdxOffset;
    disps[3] = sizeof(Int)+myRowIdxOffset;
    disps[4] = myNzValOffset;
    disps[5] = sizeof(Int)+myNzValOffset;



#if ( _DEBUGlevel_ >= 1 )
    char msg[200];
    char * tmp = msg;
    tmp += sprintf(tmp,"P%d ",mpirank);
    for(Int i = 0; i<6; ++i){
      if(i==5)
        tmp += sprintf(tmp, "%d [%d - %d] | ",i,disps[i],disps[i]+blklens[i]*sizeof(F));
      else
        tmp += sprintf(tmp, "%d [%d - %d] | ",i,disps[i],disps[i]+blklens[i]*sizeof(Int));
    }
    tmp += sprintf(tmp,"\n");
    printf("%s",msg);
#endif




    MPI_Type_create_struct(6, blklens, disps, types, &filetype);
    MPI_Type_commit(&filetype);

    /* create memory type */
    Int np1 = pspmat.size+1;
    MPI_Get_address( (void *)&np1,  &disps[0]);
    MPI_Get_address(colptrChunk.Data(), &disps[1]);
    MPI_Get_address( (void *)&pspmat.nnz,  &disps[2]);
    MPI_Get_address((void *)pspmat.rowindLocal.Data(),  &disps[3]);
    MPI_Get_address( (void *)&pspmat.nnz,  &disps[4]);
    MPI_Get_address((void *)pspmat.nzvalLocal.Data(),   &disps[5]);

    MPI_Type_create_struct(6, blklens, disps, types, &memtype);
    MPI_Type_commit(&memtype);



    /* set file view */
    err = MPI_File_set_view(fout, 0, MPI_BYTE, filetype, "native",MPI_INFO_NULL);

    /* everyone writes their own row offsets, columns, and 
     * data with one big noncontiguous write (in memory and 
     * file)
     */
    err = MPI_File_write_all(fout, MPI_BOTTOM, 1, memtype, &status);

    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);





    MPI_Barrier( comm );

    MPI_File_close(&fout);


    return ;
  }        // -----  end of function ParaWriteDistSparseMatrix  ----- 


} // namespace scales

#endif // _SPARSE_MATRIX_IMPL_HPP_
