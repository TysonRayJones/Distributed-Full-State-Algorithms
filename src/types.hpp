
#ifndef TYPES_HPP
#define TYPES_HPP


#include <complex>
#include <vector>
#include <assert.h>



/*
 * interface for specifying Pauli tensors and gadgets
 */

enum PauliOperator {
    I, X, Y, Z
};



/*
 * numerical precision config 
 */
 
typedef double Real;
typedef unsigned int Nat;
typedef long long unsigned int Index;
#define MPI_AMP MPI_DOUBLE_COMPLEX



/*
 * local complex, array, matrix convenience types 
 */
 
typedef std::complex<Real> Amp;
typedef std::vector<Amp> AmpArray;
typedef std::vector<AmpArray> AmpMatrix;
typedef std::vector<Nat> NatArray;
typedef std::vector<AmpMatrix> MatrixArray;
typedef std::vector<Real> RealArray;



/*
 * operator overloads for array and matrix
 */

static AmpMatrix getZeroMatrix(Index dim) {
    assert( dim > 0 );
    
    AmpMatrix matrix = AmpMatrix(dim);
    for (Index r=0; r<dim; r++)
        matrix[r] = AmpArray(dim, 0);
    return matrix;
}


static AmpMatrix getIdentityMatrix(Index dim) {
    assert( dim > 0 );
    
    AmpMatrix matrix = getZeroMatrix(dim);
    for (Index r=0; r<dim; r++)
        matrix[r][r] = 1;
    return matrix;
}


static AmpMatrix getConjugateMatrix(AmpMatrix matrOrig) {
    AmpMatrix matrConj = getZeroMatrix(matrOrig.size());
    for (Index r=0; r<matrConj.size(); r++)
        for (Index c=0; c<matrConj.size(); c++)
            matrConj[r][c] = conj(matrOrig[r][c]);
    return matrConj;
}


static AmpMatrix getDaggerMatrix(AmpMatrix matrOrig) {
    AmpMatrix matrDag = getZeroMatrix(matrOrig.size());
    for (Index r=0; r<matrDag.size(); r++)
        for (Index c=0; c<matrDag.size(); c++)
            matrDag[r][c] = conj(matrOrig[c][r]);
    return matrDag;
}


static AmpMatrix operator * (const Amp& f, const AmpMatrix& m) {
    AmpMatrix out = m;
    for (Nat r=0; r<out.size(); r++)
        for (Nat c=0; c<out.size(); c++)
            out[r][c] *= f;
    return out;
}


static AmpMatrix operator * (const AmpMatrix& m1, const AmpMatrix& m2) {
    assert( m1.size() == m2.size() );
    
    AmpMatrix prod = getZeroMatrix(m1.size());
    for (Nat r=0; r<m1.size(); r++)
        for (Nat c=0; c<m1.size(); c++)
            for (Nat k=0; k<m1.size(); k++)
                prod[r][c] += m1[r][k] * m2[k][c];
    return prod;
}


static AmpMatrix operator + (const AmpMatrix& m1, const AmpMatrix& m2) {
    assert( m1.size() == m2.size() );
    
    AmpMatrix res = m1;
    for (Nat r=0; r<m2.size(); r++)
        for (Nat c=0; c<m2.size(); c++)
            res[r][c] += m2[r][c];
    return res;
}


static AmpMatrix operator % (const AmpMatrix& a, const AmpMatrix& b) {
    
    // Kronecker product
    AmpMatrix prod = getZeroMatrix(a.size() * b.size());
    for (Index r=0; r<b.size(); r++)
        for (Index c=0; c<b.size(); c++)
            for (Index i=0; i<a.size(); i++)
                for (Index j=0; j<a.size(); j++)
                    prod[r+b.size()*i][c+b.size()*j] = a[i][j] * b[r][c];
    return prod;
}


static AmpArray operator * (const AmpMatrix& m, const AmpArray& v) {
    assert( m.size() == v.size() );
    
    AmpArray prod = AmpArray(v.size());
    for (Nat r=0; r<v.size(); r++)
        for (Nat c=0; c<v.size(); c++)
            prod[r] += m[r][c] * v[c];
    return prod;
}



#endif // TYPES_HPP