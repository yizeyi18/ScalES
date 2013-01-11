/// @file distvec.hpp
/// @brief Implementation of general purpose parallel vectors.
/// @author Lexing Ying and Lin Lin
/// @date 2013-01-09
#ifndef _DISTVEC_IMPL_HPP_
#define _DISTVEC_IMPL_HPP_

#include "environment.hpp"
#include "utility.hpp"
#include "distvec_decl.hpp"

namespace dgdft{

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::Insert(Key key, Data& dat)
{
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();
	iA( prtn_.Owner(key)==mpirank); 
	typename std::map<Key,Data>::iterator mi=lclmap_.find(key);  
	lclmap_[key] = dat;
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Data& DistVec<Key,Data,Partition>::Access(Key key)
{
	typename std::map<Key,Data>::iterator mi=lclmap_.find(key);
	iA(mi!=lclmap_.end());
	return (*mi).second;
}


//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::GetBegin( Int (*e2ps)(Key,Data&,std::vector<Int>&), const std::vector<Int>& mask )
{
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();
	//---------
	snbvec_.resize(mpisize,0);  for(Int k=0; k<mpisize; k++) snbvec_[k] = 0;
	rnbvec_.resize(mpisize,0);  for(Int k=0; k<mpisize; k++) rnbvec_[k] = 0;
	sbufvec_.resize(mpisize);
	rbufvec_.resize(mpisize);
	reqs_ = (MPI_Request *) malloc(2*mpisize * sizeof(MPI_Request)); iA(reqs_!=NULL);
	stats_ = (MPI_Status *) malloc(2*mpisize * sizeof(MPI_Status));
	//reqs_.resize(2*mpisize);
	//stats_.resize(2*mpisize);
	//---------
	std::vector<std::ostringstream*> ossvec(mpisize);  for(Int k=0; k<mpisize; k++)  { ossvec[k] = new std::ostringstream(); iA(ossvec[k]!=NULL); }
	//1. serialize
	for(typename std::map<Key,Data>::iterator mi=lclmap_.begin(); mi!=lclmap_.end(); mi++) {
		Key key = (*mi).first;
		const Data& dat = (*mi).second;
		if(prtn_.Owner(key)==mpirank) {
			//ASK QUESTIONS
			std::vector<Int> pids;	  Int res = (*e2ps)((*mi).first, (*mi).second, pids);
			for(Int i=0; i<pids.size(); i++) {
				Int k = pids[i];
				if(k!=mpirank) { //DO NOT SEND TO MYSELF
					iC( serialize(key, *(ossvec[k]), mask) );
					iC( serialize(dat, *(ossvec[k]), mask) );
					snbvec_[k]++; 
				}
			}
		}
	}
	// to vector
	for(Int k=0; k<mpisize; k++) {
		std::string tmp( ossvec[k]->str() );
		sbufvec_[k].clear();
		sbufvec_[k].insert(sbufvec_[k].end(), tmp.begin(), tmp.end());
	}
	for(Int k=0; k<mpisize; k++) {	delete ossvec[k];	ossvec[k] = NULL;  }
	//2. all th sendsize of the message
	std::vector<Int> sszvec(mpisize,0);
	for(Int k=0; k<mpisize; k++)
		sszvec[k] = sbufvec_[k].size();
	std::vector<Int> sifvec(2*mpisize,0);
	for(Int k=0; k<mpisize; k++) {
		sifvec[2*k  ] = snbvec_[k];
		sifvec[2*k+1] = sszvec[k];
	}
	std::vector<Int> rifvec(2*mpisize, 0);
	iC( MPI_Alltoall( (void*)&(sifvec[0]), 2, MPI_INT, (void*)&(rifvec[0]), 2, MPI_INT, comm_ ) );
	std::vector<Int> rszvec(mpisize,0);
	for(Int k=0; k<mpisize; k++) {
		rnbvec_[k] = rifvec[2*k  ];
		rszvec[k] = rifvec[2*k+1];
	}
	//3. allocate space, send and receive
	for(Int k=0; k<mpisize; k++)
		rbufvec_[k].resize(rszvec[k]);
	for(Int k=0; k<mpisize; k++) {
		iC( MPI_Irecv( (void*)&(rbufvec_[k][0]), rszvec[k], MPI_BYTE, k, 0, comm_, &reqs_[2*k] ) );
		iC( MPI_Isend( (void*)&(sbufvec_[k][0]), sszvec[k], MPI_BYTE, k, 0, comm_, &reqs_[2*k+1] ) );
	}
	iC( MPI_Barrier(comm_) );
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::GetBegin(std::vector<Key>& keyvec, const std::vector<Int>& mask)
{
	time_t t0(0), t1(0);

	t0 = time(0);

	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();
	//---------
	snbvec_.resize(mpisize,0);  for(Int k=0; k<mpisize; k++) snbvec_[k] = 0;
	rnbvec_.resize(mpisize,0);  for(Int k=0; k<mpisize; k++) rnbvec_[k] = 0;
	sbufvec_.resize(mpisize);
	rbufvec_.resize(mpisize);
	reqs_ = (MPI_Request *) malloc(2*mpisize * sizeof(MPI_Request)); iA(reqs_!=NULL);
	stats_ = (MPI_Status *) malloc(2*mpisize * sizeof(MPI_Status)); iA(stats_!=NULL);
	//reqs_.resize(2*mpisize);
	//stats_.resize(2*mpisize);
	//1. go thrw the keyvec to partition them among other procs
	std::vector< std::vector<Key> > skeyvec(mpisize);
	for(Int i=0; i<keyvec.size(); i++) {
		Key key = keyvec[i];
		Int owner = prtn_.Owner(key);
		if(owner!=mpirank)
			skeyvec[owner].push_back(key);
	}
	//2. setdn receive size of keyvec
	std::vector<Int> sszvec(mpisize);
	std::vector<Int> rszvec(mpisize);
	for(Int k=0; k<mpisize; k++)
		sszvec[k] = skeyvec[k].size();
	iC( MPI_Alltoall( (void*)&(sszvec[0]), 1, MPI_INT, (void*)&(rszvec[0]), 1, MPI_INT, comm_ ) );
	//3. allocate space for the keys, send and receive
	std::vector< std::vector<Key> > rkeyvec(mpisize);
	for(Int k=0; k<mpisize; k++)
		rkeyvec[k].resize(rszvec[k]);
	{
		//std::vector<MPI_Request> reqs_;
		//std::vector<MPI_Status> stats_;
		MPI_Request *reqs_;
		MPI_Status  *stats_;
		reqs_ = (MPI_Request *) malloc(2*mpisize * sizeof(MPI_Request)); iA(reqs_!=NULL);
		stats_ = (MPI_Status *) malloc(2*mpisize * sizeof(MPI_Status)); iA(stats_!=NULL);
		for(Int k=0; k<mpisize; k++) {
			iC( MPI_Irecv( (void*)&(rkeyvec[k][0]), rszvec[k]*sizeof(Key), MPI_BYTE, k, 0, comm_, &reqs_[2*k] ) );
			iC( MPI_Isend( (void*)&(skeyvec[k][0]), sszvec[k]*sizeof(Key), MPI_BYTE, k, 0, comm_, &reqs_[2*k+1] ) );
		}
		iC( MPI_Waitall(2*mpisize, &(reqs_[0]), &(stats_[0])) );
		free(reqs_); reqs_=NULL;
		free(stats_); stats_=NULL;
	}
	skeyvec.clear(); //save space
	//4. prepare the streams
	std::vector<std::ostringstream*> ossvec(mpisize);  for(Int k=0; k<mpisize; k++)	{ ossvec[k] = new std::ostringstream(); iA(ossvec[k]!=NULL); }
	for(Int k=0; k<mpisize; k++) {
		for(Int g=0; g<rkeyvec[k].size(); g++) {
			Key curkey = rkeyvec[k][g];
			typename std::map<Key,Data>::iterator mi = lclmap_.find(curkey);	  iA( mi!=lclmap_.end() );
			iA( prtn_.Owner(curkey)==mpirank );
			Key key = (*mi).first;
			const Data& dat = (*mi).second;
			iC( serialize(key, *(ossvec[k]), mask) );
			iC( serialize(dat, *(ossvec[k]), mask) );
			snbvec_[k]++; 
		}
	}
	// to vector
	for(Int k=0; k<mpisize; k++) {
		std::string tmp( ossvec[k]->str() );
		sbufvec_[k].clear();
		sbufvec_[k].insert(sbufvec_[k].end(), tmp.begin(), tmp.end());
	}
	for(Int k=0; k<mpisize; k++) {	delete ossvec[k];	ossvec[k] = NULL;  }
	//5. all th sendsize of the message
	for(Int k=0; k<mpisize; k++)
		sszvec[k] = sbufvec_[k].size();
	std::vector<Int> sifvec(2*mpisize,0);
	for(Int k=0; k<mpisize; k++) {
		sifvec[2*k  ] = snbvec_[k];
		sifvec[2*k+1] = sszvec[k];
	}
	std::vector<Int> rifvec(2*mpisize, 0);
	iC( MPI_Alltoall( (void*)&(sifvec[0]), 2, MPI_INT, (void*)&(rifvec[0]), 2, MPI_INT, comm_ ) );
	for(Int k=0; k<mpisize; k++) {
		rnbvec_[k] = rifvec[2*k  ];
		rszvec[k] = rifvec[2*k+1];
	}
	t1 = time(0);
	//6. allocate space, send and receive
	for(Int k=0; k<mpisize; k++)
		rbufvec_[k].resize(rszvec[k]);
	for(Int k=0; k<mpisize; k++) {
		iC( MPI_Irecv( (void*)&(rbufvec_[k][0]), rszvec[k], MPI_BYTE, k, 0, comm_, &reqs_[2*k] ) );
		iC( MPI_Isend( (void*)&(sbufvec_[k][0]), sszvec[k], MPI_BYTE, k, 0, comm_, &reqs_[2*k+1] ) );
	}
	iC( MPI_Barrier(comm_) );
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::GetEnd( const std::vector<Int>& mask )
{
	time_t t0(0), t1(0);

	t0 = time(0);
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();
	
	iC( MPI_Waitall(2*mpisize, &(reqs_[0]), &(stats_[0])) );
	free(reqs_); reqs_=NULL;
	free(stats_); stats_=NULL;
	sbufvec_.clear(); //save space
	iC( MPI_Barrier(comm_) );

	t0 = time(0);
	//4. write back
	//to stream
	std::vector<std::istringstream*> issvec(mpisize);  for(Int k=0; k<mpisize; k++)	{ issvec[k] = new std::istringstream(); iA(issvec[k]!=NULL); }
	for(Int k=0; k<mpisize; k++) {
		std::string tmp(rbufvec_[k].begin(), rbufvec_[k].end());
		issvec[k]->str(tmp);
	}
	rbufvec_.clear(); //save space
	iC( MPI_Barrier(comm_) );

	for(Int k=0; k<mpisize; k++) {
		for(Int i=0; i<rnbvec_[k]; i++) {
			Key key;  deserialize(key, *(issvec[k]), mask);
			typename std::map<Key,Data>::iterator mi=lclmap_.find(key);
			if(mi==lclmap_.end()) { //do not exist
				Data dat;		deserialize(dat, *(issvec[k]), mask);
				lclmap_[key] = dat;
			} else { //exist already
				deserialize((*mi).second, *(issvec[k]), mask);
			}
		}
	}
	for(Int k=0; k<mpisize; k++) {	delete issvec[k];	issvec[k] = NULL;  }
	iC( MPI_Barrier(comm_) );

	t1 = time(0);

	iC( MPI_Barrier(comm_) );
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::PutBegin(std::vector<Key>& keyvec, const std::vector<Int>& mask)
{
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();
	//---------
	snbvec_.resize(mpisize,0);  for(Int k=0; k<mpisize; k++) snbvec_[k] = 0;
	rnbvec_.resize(mpisize,0);  for(Int k=0; k<mpisize; k++) rnbvec_[k] = 0;
	sbufvec_.resize(mpisize);
	rbufvec_.resize(mpisize);
	reqs_ = (MPI_Request *) malloc(2*mpisize * sizeof(MPI_Request)); iA(reqs_!=NULL);
	stats_ = (MPI_Status *) malloc(2*mpisize * sizeof(MPI_Status)); iA(stats_!=NULL);
	//reqs_.resize(2*mpisize);
	//stats_.resize(2*mpisize);
	//1.
	std::vector<std::ostringstream*> ossvec(mpisize);  for(Int k=0; k<mpisize; k++)	{ ossvec[k] = new std::ostringstream(); iA(ossvec[k]!=NULL); }
	//1. go thrw the keyvec to partition them among other procs
	for(Int i=0; i<keyvec.size(); i++) {
		Key key = keyvec[i];
		Int k = prtn_.Owner(key); //the owner
		if(k!=mpirank) {
			typename std::map<Key,Data>::iterator mi = lclmap_.find(key);
			iA( mi!=lclmap_.end() );	  iA( key==(*mi).first );
			Data& dat = (*mi).second;
			iC( serialize(key, *(ossvec[k]), mask) );
			iC( serialize(dat, *(ossvec[k]), mask) );
			snbvec_[k]++;
		}
	}
	//2. to std::vector
	for(Int k=0; k<mpisize; k++) {
		std::string tmp( ossvec[k]->str() );
		sbufvec_[k].clear();
		sbufvec_[k].insert(sbufvec_[k].end(), tmp.begin(), tmp.end());
	}
	for(Int k=0; k<mpisize; k++) {	delete ossvec[k];	ossvec[k] = NULL;  }
	//3. get size
	std::vector<Int> sszvec(mpisize);
	for(Int k=0; k<mpisize; k++)
		sszvec[k] = sbufvec_[k].size();
	std::vector<Int> sifvec(2*mpisize,0);
	for(Int k=0; k<mpisize; k++) {
		sifvec[2*k  ] = snbvec_[k];
		sifvec[2*k+1] = sszvec[k];
	}
	std::vector<Int> rifvec(2*mpisize, 0);
	iC( MPI_Alltoall( (void*)&(sifvec[0]), 2, MPI_INT, (void*)&(rifvec[0]), 2, MPI_INT, comm_ ) );
	std::vector<Int> rszvec(mpisize,0);
	for(Int k=0; k<mpisize; k++) {
		rnbvec_[k] = rifvec[2*k  ];
		rszvec[k] = rifvec[2*k+1];
	}
	//4. allocate space, send and receive
	for(Int k=0; k<mpisize; k++)
		rbufvec_[k].resize(rszvec[k]);
	for(Int k=0; k<mpisize; k++) {
		iC( MPI_Irecv( (void*)&(rbufvec_[k][0]), rszvec[k], MPI_BYTE, k, 0, comm_, &reqs_[2*k] ) );
		iC( MPI_Isend( (void*)&(sbufvec_[k][0]), sszvec[k], MPI_BYTE, k, 0, comm_, &reqs_[2*k+1] ) );
	}
	iC( MPI_Barrier(comm_) );
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::PutEnd( const std::vector<Int>& mask, Int putmode )
{
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();

	iC( MPI_Waitall(2*mpisize, &(reqs_[0]), &(stats_[0])) );
	free(reqs_); reqs_=NULL;
	free(stats_); stats_=NULL;
	sbufvec_.clear(); //save space
	//5. go thrw the messages and write back
	std::vector<std::istringstream*> issvec(mpisize);  for(Int k=0; k<mpisize; k++)	{ issvec[k] = new std::istringstream(); iA(issvec[k]!=NULL); }
	for(Int k=0; k<mpisize; k++) {
		std::string tmp(rbufvec_[k].begin(), rbufvec_[k].end());
		issvec[k]->str(tmp);
	}
	rbufvec_.clear(); //save space
	for(Int k=0; k<mpisize; k++) {
		for(Int i=0; i<rnbvec_[k]; i++) {
			Key key;	  deserialize(key, *(issvec[k]), mask);	  iA( prtn_.Owner(key)==mpirank );
			
			typename std::map<Key,Data>::iterator mi=lclmap_.find(key);
			if(mi==lclmap_.end() ) { //DO NOT EVEN EXIST
				//----------------------------------
				Data tmp;	deserialize(tmp, *(issvec[k]), mask);
				lclmap_[key] = tmp;
			} else { //DO EXIST
				//----------------------------------
				if( putmode == PutMode::REPLACE) {
					deserialize((*mi).second, *(issvec[k]), mask);
				} else if( putmode == PutMode::COMBINE ) {
					Data tmp;	deserialize(tmp, *(issvec[k]), mask);
					combine( (*mi).second, tmp); 
				}
			}
		}
	}
	for(Int k=0; k<mpisize; k++) {	delete issvec[k];	issvec[k] = NULL;  }
	iC( MPI_Barrier(comm_) );
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::Expand(std::vector<Key>& keyvec)
{
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();
	Data dummy;
	for(Int i=0; i<keyvec.size(); i++) {
		Key key = keyvec[i];
		typename std::map<Key,Data>::iterator mi=lclmap_.find(key);
		if(mi==lclmap_.end()) {
			lclmap_[key] = dummy;
		}
	}
	return 0;
}

//--------------------------------------------
template <class Key, class Data, class Partition>
Int DistVec<Key,Data,Partition>::Discard(std::vector<Key>& keyvec)
{
	Int mpirank = this->mpirank();
	Int mpisize = this->mpisize();

	for(Int i=0; i<keyvec.size(); i++) {
		Key key = keyvec[i];
		if(prtn_.Owner(key)!=mpirank) {
			lclmap_.erase(key);
		}
	}
	return 0;
}

//-------------------
template<class Key, class Data, class Partition>
Int serialize(const DistVec<Key,Data,Partition>& pv, std::ostream& os, const std::vector<Int>& mask)
{
	serialize(pv.lclmap_, os, mask);  
	serialize(pv.prtn_, os, mask);
	return 0;
}

template<class Key, class Data, class Partition>
Int deserialize(DistVec<Key,Data,Partition>& pv, std::istream& is, const std::vector<Int>& mask)
{
	deserialize(pv.lclmap_, is, mask);
	deserialize(pv.prtn_, is, mask);
	return 0;
}

} //  namespace dgdft

#endif  // _DISTVEC_IMPL_HPP_

