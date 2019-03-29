/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#include <immintrin.h>

#include <string.h>
#include "hadamard_intrinsics_pitch.h"

///Do the Hadamard product
/**	@param[out] tabResult : table of results of tabX*tabY
 * 	@param tabX : input table
 * 	@param tabY : input table
 * 	@param nbElement : number of elements in the tables
*/
void hadamard_product(float* tabResult, const float* tabX, const float* tabY, long unsigned int nbElement){
	long unsigned int vecSize(VECTOR_ALIGNEMENT/sizeof(float));
	long unsigned int nbVec(nbElement/vecSize);
	for(long unsigned int i(0lu); i < nbVec; ++i){
		__m256 vecX = _mm256_load_ps(tabX + i*vecSize);
		__m256 vecY = _mm256_load_ps(tabY + i*vecSize);
		__m256 vecRes = _mm256_mul_ps(vecX, vecY);
		_mm256_store_ps(tabResult + i*vecSize, vecRes);
	}
}

