// Copyright (C) 2011 Technische Universitaet Muenchen
// This file is part of the preCICE project. For conditions of distribution and
// use, please see the license notice at http://www5.in.tum.de/wiki/index.php/PreCICE_License
#include "IQNILSPostProcessing.hpp"
#include "cplscheme/CouplingData.hpp"
#include "utils/Globals.hpp"
#include "tarch/la/GramSchmidt.h"
#include "tarch/la/MatrixVectorOperations.h"
#include "tarch/la/TransposedMatrix.h"
#include "mesh/Mesh.hpp"
#include "mesh/Vertex.hpp"
#include "utils/Dimensions.hpp"
#include "tarch/la/Scalar.h"
#include "io/TXTWriter.hpp"
#include "io/TXTReader.hpp"
#include "utils/MasterSlave.hpp"
#include "QRFactorization.hpp"
#include "Eigen/Dense"
#include <sys/unistd.h>

#include "tarch/tests/TestMacros.h"

#include <time.h>

//#include "utils/NumericalCompare.hpp"

namespace precice {
namespace cplscheme {
namespace impl {

// tarch::logging::Log IQNILSPostProcessing::
//       _log("precice::cplscheme::impl::IQNILSPostProcessing");

IQNILSPostProcessing:: IQNILSPostProcessing
(
  double initialRelaxation,
  int    maxIterationsUsed,
  int    timestepsReused,
  int 	 filter,
  double singularityLimit,
  std::vector<int> dataIDs,
  std::map<int,double> scalings)
:
  BaseQNPostProcessing(initialRelaxation, maxIterationsUsed, timestepsReused,
		       filter, singularityLimit, dataIDs, scalings),
  _secondaryOldXTildes(),
  _secondaryMatricesW(),
  _secondaryMatricesWBackup()
{
}

void IQNILSPostProcessing:: initialize
(
  DataMap& cplData )
{
  // do common QN post processing initialization
  BaseQNPostProcessing::initialize(cplData);

  double init = 0.0;
  // Fetch secondary data IDs, to be relaxed with same coefficients from IQN-ILS
  foreach (DataMap::value_type& pair, cplData){
	if (not utils::contained(pair.first, _dataIDs)){
	  int secondaryEntries = pair.second->values->size();
	  _secondaryOldXTildes[pair.first].append(DataValues(secondaryEntries, init));
	}
  }
}


void IQNILSPostProcessing::updateDifferenceMatrices
(
  DataMap& cplData)
{
	// Compute residuals of secondary data
	for (int id: _secondaryDataIDs){
		DataValues& secResiduals = _secondaryResiduals[id];
		PtrCouplingData data = cplData[id];
		assertion2(secResiduals.size() == data->values->size(),
				secResiduals.size(), data->values->size());
		secResiduals = *(data->values);
		secResiduals -= data->oldValues.column(0);
	}

	//if(_firstIteration && (_firstTimeStep || (_matrixCols.size() < 2))) {
	if (_firstIteration && _firstTimeStep){
		// constant relaxation: for secondary data called from base class
	}else{
		if (not _firstIteration) {
			bool columnLimitReached = getLSSystemCols() == _maxIterationsUsed;
			bool overdetermined = getLSSystemCols() <= getLSSystemRows();
			if (not columnLimitReached && overdetermined) {

				// Append column for secondary W matrices
				for (int id: _secondaryDataIDs) {
					_secondaryMatricesW[id].appendFront(_secondaryResiduals[id]);
				}
			}
			else {
				// Shift column for secondary W matrices
				for (int id: _secondaryDataIDs) {
					_secondaryMatricesW[id].shiftSetFirst(_secondaryResiduals[id]);
				}
			}

			// Compute delta_x_tilde for secondary data
			for (int id: _secondaryDataIDs) {
				DataMatrix& secW = _secondaryMatricesW[id];
				assertion2(secW.column(0).size() == cplData[id]->values->size(),
						secW.column(0).size(), cplData[id]->values->size());
				secW.column(0) = *(cplData[id]->values);
				secW.column(0) -= _secondaryOldXTildes[id];
			}
		}

		// Store x_tildes for secondary data
		for (int id: _secondaryDataIDs) {
			assertion2(_secondaryOldXTildes[id].size() == cplData[id]->values->size(),
					_secondaryOldXTildes[id].size(), cplData[id]->values->size());
			_secondaryOldXTildes[id] = *(cplData[id]->values);
		}
	}
  
  // call the base method for common update of V, W matrices
  BaseQNPostProcessing::updateDifferenceMatrices(cplData);
}


void IQNILSPostProcessing::computeUnderrelaxationSecondaryData
(
  DataMap& cplData)
{
    //Store x_tildes for secondary data
    for (int id: _secondaryDataIDs){
      assertion2(_secondaryOldXTildes[id].size() == cplData[id]->values->size(),
                 _secondaryOldXTildes[id].size(), cplData[id]->values->size());
      _secondaryOldXTildes[id] = *(cplData[id]->values);
    }

    // Perform underrelaxation with initial relaxation factor for secondary data
    for (int id: _secondaryDataIDs){
      PtrCouplingData data = cplData[id];
      DataValues& values = *(data->values);
      values *= _initialRelaxation;                   // new * omg
      DataValues& secResiduals = _secondaryResiduals[id];
      secResiduals = data->oldValues.column(0);    // old
      secResiduals *= 1.0 - _initialRelaxation;       // (1-omg) * old
      values += secResiduals;                      // (1-omg) * old + new * omg
    }
}


void IQNILSPostProcessing::computeQNUpdate
(PostProcessing::DataMap& cplData, DataValues& xUpdate)
{
	preciceTrace("computeQNUpdate()");
    using namespace tarch::la;
    preciceDebug("   Compute Newton factors");

    // Calculate QR decomposition of matrix V and solve Rc = -Qr
    DataValues __c;

    if(_filter == BaseQNPostProcessing::NOFILTER){
    	// do nothing

    /**
    }else if(_filter == BaseQNPostProcessing::QR2FILTER){
		 Eigen::MatrixXd _Q, _R;
		bool termination = false;

		while(!termination)
		{
			// copy matrix V to Eigen Matrix data type
			Eigen::MatrixXd _V(_matrixV.rows(), _matrixV.cols());
			for (int i = 0; i < _V.rows(); i++)
				for (int j = 0; j < _V.cols(); j++) {
					_V(i, j) = _matrixV(i, j);
				}
			// copy matrix W to Eigen Matrix data type
			Eigen::MatrixXd _W(_matrixW.rows(), _matrixW.cols());
			for (int i = 0; i < _W.rows(); i++)
				for (int j = 0; j < _W.cols(); j++) {
					_W(i, j) = _matrixW(i, j);
				}

			termination = true;

			// compute QR-decomposition of V
			_Q = Eigen::MatrixXd(_matrixV.rows(), _matrixV.cols());
			_R = Eigen::MatrixXd(_matrixV.cols(), _matrixV.cols());

			Eigen::VectorXd v0 = _V.col(0);
			_R(0,0) = v0.norm();
			_Q.col(0) = v0/_R(0,0);
			for( int j=1; j<_V.cols(); j++)
			{
				Eigen::VectorXd v = _V.col(j);
				double rho0 = v.norm();

				for(int i=0; i < j; i++)
				{
					Eigen::VectorXd Qci = _Q.col(i);
					_R(i,j) = Qci.dot(v);
					v = v -_R(i,j)*Qci;
				}

				// QR-filter test (if information that comes with vector v
				// is little, i.e., |v_orth| small, discard vector v.)
				double rho1 = v.norm();
				if(rho1 < _singularityLimit * rho0)
				{
					termination = false;
					removeMatrixColumn(j);
					_qrV.deleteColumn(j);

					preciceDebug("   (QR2-Filter) t="<<tSteps<<", k="<<its<<" | deleting column " << j );
					_infostream <<"   (QR2-Filter) t="<<tSteps<<", k="<<its
								<<" | deleting column " << j << std::flush<<std::endl;

					break;
				}

				// normalize
				_R(j,j) = rho1;
				_Q.col(j) = v/rho1;
			}
		}
		// copy back, Q and R
		DataMatrix Q(_Q.rows(), _Q.cols(), 0.0);
		for (int i = 0; i < _Q.rows(); i++)
			for (int j = 0; j < _Q.cols(); j++) {
				Q(i, j) = _Q(i, j);
			}
		DataMatrix R(_R.rows(), _R.cols(), 0.0);
		for (int j = 0; j < _R.cols(); j++)
			for (int i = 0; i <= j; i++) {
				R(i, j) = _R(i, j);
			}
		// compute update
		DataValues c;
		DataValues b(Q.cols(), 0.0);
		tarch::la::multiply(tarch::la::transpose(Q), _residuals, b); // = Qr
		b *= -1.0; // = -Qr
		assertion1(c.size() == 0, c.size());
		c.append(b.size(), 0.0);

		tarch::la::backSubstitution(R, b, c);
		tarch::la::multiply(_matrixW, c, xUpdate);

		preciceDebug("c = " << c);

		// Perform QN relaxation for secondary data
		for (int id: _secondaryDataIDs){
			PtrCouplingData data = cplData[id];
			DataValues& values = *(data->values);
			assertion2(_secondaryMatricesW[id].cols() == c.size(),
					_secondaryMatricesW[id].cols(), c.size());
			tarch::la::multiply(_secondaryMatricesW[id], c, values);
			assertion2(values.size() == data->oldValues.column(0).size(),
					values.size(), data->oldValues.column(0).size());
			values += data->oldValues.column(0);
			assertion2(values.size() == _secondaryResiduals[id].size(),
					values.size(), _secondaryResiduals[id].size());
			values += _secondaryResiduals[id];
		}
		return;
	*/
		/**
	}
	else if(_filter == BaseQNPostProcessing::PODFILTER){

		// copy matrix V to Eigen Matrix data type
		Eigen::MatrixXd _V(_matrixV.rows(), _matrixV.cols());
		for (int i = 0; i < _V.rows(); i++)
			for (int j = 0; j < _V.cols(); j++) {
				_V(i, j) = _matrixV(i, j);
			}
		// copy matrix W to Eigen Matrix data type
		Eigen::MatrixXd _W(_matrixW.rows(), _matrixW.cols());
		for (int i = 0; i < _W.rows(); i++)
			for (int j = 0; j < _W.cols(); j++) {
				_W(i, j) = _matrixW(i, j);
			}

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(_V, Eigen::ComputeFullV);
		Eigen::VectorXd sigma = svd.singularValues();
		Eigen::MatrixXd phi = svd.matrixV();

		//assertion2(sigma.size() == _matrixV.cols(), sigma.size(), _matrixV.cols);
		assertion2(phi.rows() == _V.cols(), phi.rows(), _V.cols());
		assertion2(phi.rows() == _W.cols(), phi.rows(), _W.cols());

		int k = _V.cols();
		double lambda_1 = sigma(0)*sigma(0)/phi.rows();
		for(int i = 1; i < sigma.size(); i++)
		{
			double lambda_i = sigma(i)*sigma(i)/phi.rows();
			if(lambda_i/lambda_1 <= _singularityLimit)
			{
				k = i;

				// print
				preciceDebug("   (POD-Filter) t="<<tSteps<<", k="<<its
							<<" | truncating matrices VX, WX after the first " << k
							<<"columns. Discarded columns: "<< _V.cols()-k);
				_infostream << "   (POD-Filter) t="<<tSteps<<", k="<<its
							<<" | truncating matrices VX, WX after the first " << k
							<<"columns. Discarded columns: "<< _V.cols()-k<< std::flush<<std::endl;
				// debugging information, can be removed
				deletedColumns += _V.rows()-k;
				break;
			}
		}

		// compute V_til = V*phi and W_til = W*phi
		_V = _V*phi;
		_W = _W*phi;

		// truncate
		_V.conservativeResize(_V.rows(), k);
		_W.conservativeResize(_W.rows(), k);


		DataMatrix Vcopy(_V.rows(), _V.cols(), 0.0);
		for (int i = 0; i < _V.rows(); i++)
				for (int j = 0; j < _V.cols(); j++) {
					Vcopy(i, j) = _V(i, j);
				}
		DataMatrix Wcopy(_W.rows(), _W.cols(), 0.0);
		for (int i = 0; i < _W.rows(); i++)
				for (int j = 0; j < _W.cols(); j++) {
					Wcopy(i, j) = _W(i, j);
				}
		DataMatrix Q(Vcopy.rows(), Vcopy.cols(), 0.0);
		DataMatrix R(Vcopy.cols(), Vcopy.cols(), 0.0);
		tarch::la::modifiedGramSchmidt(Vcopy, Q, R);

		DataValues c;
		DataValues b(Q.cols(), 0.0);
		tarch::la::multiply(tarch::la::transpose(Q), _residuals, b); // = Qr
		b *= -1.0; // = -Qr
		assertion1(c.size() == 0, c.size());
		c.append(b.size(), 0.0);

		tarch::la::backSubstitution(R, b, c);
		tarch::la::multiply(Wcopy, c, xUpdate);

		return;
		*/

	}else{
		// do: filtering of least-squares system to maintain good conditioning
		std::vector<int> delIndices(0);
		_qrV.applyFilter(_singularityLimit, delIndices, _matrixV);
		// start with largest index (as V,W matrices are shrinked and shifted
		for(int i = delIndices.size()-1; i >= 0; i--){
			preciceDebug("   Removing linear dependent column " << delIndices[i]);
			removeMatrixColumn(delIndices[i]);
			_infostream<<"[QR-dec] - removing linear dependent column "<<delIndices[i]<<"\n"<<std::flush;
		}
		assertion2(_matrixV.cols() == _qrV.cols(), _matrixV.cols(), _qrV.cols());
	}


	// for master-slave mode and procs with no vertices,
	// qrV.cols() = getLSSystemCols() and _qrV.rows() = 0
	Matrix __Qt(_qrV.cols(), _qrV.rows(), 0.0);
	Matrix __R(_qrV.cols(), _qrV.cols(), 0.0);

	auto q = _qrV.matrixQ();
	for(int i = 0; i<q.rows(); i++)
	for(int j = 0; j<q.cols(); j++)
	{
		__Qt(j,i) = q(i,j);
	}

	if(!_hasNodesOnInterface){
	assertion2(_qrV.cols() == getLSSystemCols(), _qrV.cols(), getLSSystemCols());
	assertion1(_qrV.rows() == 0, _qrV.rows());
	assertion1(__Qt.size() == 0, __Qt.size());
	}

	auto r = _qrV.matrixR();
	for(int i = 0; i<r.rows(); i++)
	for(int j = 0; j<r.cols(); j++)
	{
		__R(i,j) = r(i,j);
	}

	DataValues _local_b(_qrV.cols(), 0.0);
	DataValues _global_b;

	multiply(__Qt, _residuals, _local_b);
	_local_b *= -1.0; // = -Qr

	assertion1(__c.size() == 0, __c.size());

	/**
	 * compute rhs Q^T*res in parallel
	 * TODO: implement all-reduce
	 */
	if (not utils::MasterSlave::_masterMode && not utils::MasterSlave::_slaveMode) {
		assertion2(__Qt.rows() == getLSSystemCols(), __Qt.rows(), getLSSystemCols());
		__c.append(_local_b.size(), 0.0);
		backSubstitution(__R, _local_b, __c);
	}else{

	   assertion(utils::MasterSlave::_communication.get() != nullptr);
	   assertion(utils::MasterSlave::_communication->isConnected());

	   if(_hasNodesOnInterface)  assertion2(__Qt.rows() == getLSSystemCols(), __Qt.rows(), getLSSystemCols());

	   // reserve memory for c
	   __c.append(_local_b.size(), 0.0);

	  if(utils::MasterSlave::_slaveMode){
		  assertion2(_local_b.size() == getLSSystemCols(), _local_b.size(), getLSSystemCols());
		  utils::MasterSlave::_communication->send(&_local_b(0), _local_b.size(), 0);
	  }
	  if(utils::MasterSlave::_masterMode){
		assertion1(_global_b.size() == 0, _global_b.size());
		assertion2(_local_b.size() == getLSSystemCols(), _local_b.size(), getLSSystemCols());

		_global_b.append(_local_b.size(), 0.0);
		_global_b += _local_b;

		for(int rankSlave = 1; rankSlave <  utils::MasterSlave::_size; rankSlave++){
			utils::MasterSlave::_communication->receive(&_local_b(0), _local_b.size(), rankSlave);
			_global_b += _local_b;
		}
		// backsubstitution only in master
		backSubstitution(__R, _global_b, __c);
	  }

	  // broadcast coefficients c to all slaves
	  utils::MasterSlave::broadcast(&__c(0), __c.size());
	}

	preciceDebug("   Apply Newton factors");
	// compute x updates from W and coefficients c, i.e, xUpdate = c*W
	multiply(_matrixW, __c, xUpdate);

	preciceDebug("c = " << __c);


    /**
     *  perform QN-Update step for the secondary Data
     */

	// If the previous time step converged within one single iteration, nothing was added
	// to the LS system matrices and they need to be restored from the backup at time T-2
    if (not _firstTimeStep && (getLSSystemCols() < 1) && (_timestepsReused == 0)) {
		preciceDebug("   Last time step converged after one iteration. Need to restore the secondaryMatricesW from backup.");
		_secondaryMatricesW = _secondaryMatricesWBackup;
	}

	// Perform QN relaxation for secondary data
	for (int id: _secondaryDataIDs){
	  PtrCouplingData data = cplData[id];
	  DataValues& values = *(data->values);
	  assertion2(_secondaryMatricesW[id].cols() == __c.size(),
				 _secondaryMatricesW[id].cols(), __c.size());
	  multiply(_secondaryMatricesW[id], __c, values);
	  assertion2(values.size() == data->oldValues.column(0).size(),
				 values.size(), data->oldValues.column(0).size());
	  values += data->oldValues.column(0);
	  assertion2(values.size() == _secondaryResiduals[id].size(),
				 values.size(), _secondaryResiduals[id].size());
	  values += _secondaryResiduals[id];
	}

	// pending deletion: delete old secondaryMatricesW
	if (_firstIteration && _timestepsReused == 0) {
		// save current secondaryMatrix data in case the coupling for the next time step will terminate
		// after the first iteration (no new data, i.e., V = W = 0)
		if(getLSSystemCols() > 0){
			_secondaryMatricesWBackup = _secondaryMatricesW;
		}
		for (int id: _secondaryDataIDs){
			_secondaryMatricesW[id].clear();
		}
	}
}


//void IQNILSPostProcessing::computeQNUpdate
//(PostProcessing::DataMap& cplData, DataValues& xUpdate)
//{
//  preciceTrace("computeQNUpdate()");
//    using namespace tarch::la;
//
//    // Calculate QR decomposition of matrix V and solve Rc = -Qr
//    DataValues __c;
//    bool linearDependence = true;
//    while (linearDependence){
//      preciceDebug("   Compute Newton factors");
//      linearDependence = false;
//
//      Matrix __R(_qrV.cols(), _qrV.cols(), 0.0);
//      auto r = _qrV.matrixR();
//        for(int i = 0; i<r.rows(); i++)
//          for(int j = 0; j<r.cols(); j++)
//          {
//            __R(i,j) = r(i,j);
//          }
//      if (getLSSystemCols() > 1){
//        for (int i=0; i < __R.rows(); i++){
//          if (std::fabs(__R(i,i)) < _singularityLimit * r.norm()){
//        	preciceDebug("   Removing linear dependent column " << i);
//        	_infostream<<"[QR-dec] - removing linear dependent column "<<i<<"\n"<<std::flush;
//            linearDependence = true;
//            removeMatrixColumn(i);
//          }
//        }
//      }
//      if (not linearDependence){
//
//	preciceDebug("   Apply Newton factors");
//
//	// for master-slave mode and procs with no vertices,
//	// grV.cols() = getLSSystemCols() and _qrV.rows() = 0
//	Matrix __Qt(_qrV.cols(), _qrV.rows(), 0.0);
//
//	auto q = _qrV.matrixQ();
//	for(int i = 0; i<q.rows(); i++)
//	  for(int j = 0; j<q.cols(); j++)
//	  {
//	    __Qt(j,i) = q(i,j);
//	  }
//
//	if(!_hasNodesOnInterface){
//		assertion2(_qrV.cols() == getLSSystemCols(), _qrV.cols(), getLSSystemCols());
//		assertion1(_qrV.rows() == 0, _qrV.rows());
//		assertion1(__Qt.size() == 0, __Qt.size());
//	}
//
//	auto r = _qrV.matrixR();
//	for(int i = 0; i<r.rows(); i++)
//	  for(int j = 0; j<r.cols(); j++)
//	  {
//	    __R(i,j) = r(i,j);
//	  }
//
//		DataValues _local_b(_qrV.cols(), 0.0);
//		DataValues _global_b;
//
//		multiply(__Qt, _residuals, _local_b);
//		_local_b *= -1.0; // = -Qr
//
//		assertion1(__c.size() == 0, __c.size());
//
//		/**
//		 * compute rhs Q^T*res in parallel
//		 * TODO: implement all-reduce
//		 */
//		if (not utils::MasterSlave::_masterMode && not utils::MasterSlave::_slaveMode) {
//			assertion2(__Qt.rows() == getLSSystemCols(), __Qt.rows(), getLSSystemCols());
//			__c.append(_local_b.size(), 0.0);
//		  	backSubstitution(__R, _local_b, __c);
//		}else{
//
//		   assertion(utils::MasterSlave::_communication.get() != nullptr);
//		   assertion(utils::MasterSlave::_communication->isConnected());
//
//		   if(_hasNodesOnInterface)  assertion2(__Qt.rows() == getLSSystemCols(), __Qt.rows(), getLSSystemCols());
//
//		   // reserve memory for c
//		   __c.append(_local_b.size(), 0.0);
//
//		  if(utils::MasterSlave::_slaveMode){
//			  assertion2(_local_b.size() == getLSSystemCols(), _local_b.size(), getLSSystemCols());
//			  utils::MasterSlave::_communication->send(&_local_b(0), _local_b.size(), 0);
//		  }
//		  if(utils::MasterSlave::_masterMode){
//			assertion1(_global_b.size() == 0, _global_b.size());
//			assertion2(_local_b.size() == getLSSystemCols(), _local_b.size(), getLSSystemCols());
//
//			_global_b.append(_local_b.size(), 0.0);
//			_global_b += _local_b;
//
//			for(int rankSlave = 1; rankSlave <  utils::MasterSlave::_size; rankSlave++){
//				utils::MasterSlave::_communication->receive(&_local_b(0), _local_b.size(), rankSlave);
//				_global_b += _local_b;
//			}
//			// backsubstitution only in master
//			backSubstitution(__R, _global_b, __c);
//		  }
//
//		  // broadcast coefficients c to all slaves
//		  utils::MasterSlave::broadcast(&__c(0), __c.size());
//		}
//
//		// compute x updates from W and coefficients c, i.e, xUpdate = c*W
//		multiply(_matrixW, __c, xUpdate);
//
//		preciceDebug("c = " << __c);
//		//_infostream<<"c = "<<__c<<"\n"<<std::flush;
//      }
//    }
//
//
//    /**
//     *  perform QN-Update step for the secondary Data
//     */
//
//	// If the previous time step converged within one single iteration, nothing was added
//	// to the LS system matrices and they need to be restored from the backup at time T-2
//    if (not _firstTimeStep && (getLSSystemCols() < 1) && (_timestepsReused == 0)) {
//		preciceDebug("   Last time step converged after one iteration. Need to restore the secondaryMatricesW from backup.");
//		_secondaryMatricesW = _secondaryMatricesWBackup;
//	}
//
//	// Perform QN relaxation for secondary data
//	for (int id: _secondaryDataIDs){
//	  PtrCouplingData data = cplData[id];
//	  DataValues& values = *(data->values);
//	  assertion2(_secondaryMatricesW[id].cols() == __c.size(),
//				 _secondaryMatricesW[id].cols(), __c.size());
//	  multiply(_secondaryMatricesW[id], __c, values);
//	  assertion2(values.size() == data->oldValues.column(0).size(),
//				 values.size(), data->oldValues.column(0).size());
//	  values += data->oldValues.column(0);
//	  assertion2(values.size() == _secondaryResiduals[id].size(),
//				 values.size(), _secondaryResiduals[id].size());
//	  values += _secondaryResiduals[id];
//	}
//
//	// pending deletion: delete old secondaryMatricesW
//	if (_firstIteration && _timestepsReused == 0) {
//		// save current secondaryMatrix data in case the coupling for the next time step will terminate
//		// after the first iteration (no new data, i.e., V = W = 0)
//		if(getLSSystemCols() > 0){
//			_secondaryMatricesWBackup = _secondaryMatricesW;
//		}
//		for (int id: _secondaryDataIDs){
//			_secondaryMatricesW[id].clear();
//		}
//	}
//}


void IQNILSPostProcessing:: specializedIterationsConverged
(
   DataMap & cplData)
{
  
  if (_matrixCols.front() == 0){ // Did only one iteration
    _matrixCols.pop_front(); 
  }

  if (_timestepsReused == 0){
	// pending deletion of secondaryMatricesW
  }
  else if ((int)_matrixCols.size() > _timestepsReused){
	int toRemove = _matrixCols.back();
	for (int id: _secondaryDataIDs){
	  DataMatrix& secW = _secondaryMatricesW[id];
	  assertion3(secW.cols() > toRemove, secW, toRemove, id);
	  for (int i=0; i < toRemove; i++){
		secW.remove(secW.cols() - 1);
	  }
	}
  }
}


void IQNILSPostProcessing:: removeMatrixColumn
(
  int columnIndex)
{
  assertion(_matrixV.cols() > 1);
  // remove column from secondary Data Matrix W
  for (int id: _secondaryDataIDs){
	 _secondaryMatricesW[id].remove(columnIndex);
   }

	BaseQNPostProcessing::removeMatrixColumn(columnIndex);
}

}}} // namespace precice, cplscheme, impl
