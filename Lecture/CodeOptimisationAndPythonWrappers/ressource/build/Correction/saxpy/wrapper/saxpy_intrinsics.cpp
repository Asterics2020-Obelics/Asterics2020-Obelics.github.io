/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#include <immintrin.h>

#include <string.h>
#include "saxpy_intrinsics.h"

///Do the saxpy
/**	@param[out] tabResult : table of results of tabX*tabY
 * 	@param scal : multiplication scalar (a)
 * 	@param tabX : input table
 * 	@param tabY : input table
 * 	@param nbElement : number of elements in the tables
*/
void saxpy(float* tabResult, float scal, const float * tabX, const float* tabY, long unsigned int nbElement){
	__m256 vecScal = _mm256_broadcast_ss(&scal);
	long unsigned int vecSize(VECTOR_ALIGNEMENT/sizeof(float));
	long unsigned int nbVec(nbElement/vecSize);
	for(long unsigned int i(0lu); i < nbVec; ++i){
		// tabResult = scal*tabX + tabY
		__m256 vecX = _mm256_load_ps(tabX + i*vecSize);
		__m256 vecAX = _mm256_mul_ps(vecX, vecScal);
		__m256 vecY = _mm256_load_ps(tabY + i*vecSize);
		__m256 vecRes = _mm256_add_ps(vecAX, vecY);
		_mm256_store_ps(tabResult + i*vecSize, vecRes);
	}
}

