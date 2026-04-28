#include <emscripten.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================
// 3D linear-elastic FEA kernel — 8-node hexahedral elements
// CSR sparse storage, penalty-method BCs
// ============================================================

static int nNodes = 0, nElems = 0, nDof = 0, nnz = 0;
static int g_nx = 0, g_ny = 0, g_nz = 0;

static std::vector<double> nodeCoords;   // 3*nNodes
static std::vector<int>    elemConn;     // 8*nElems
static std::vector<int>    csrRowPtr;
static std::vector<int>    csrColIdx;
static std::vector<double> csrValues;
static std::vector<double> rhsVec;
static std::vector<double> solVec;
static std::vector<double> stressData;   // 6*nElems (xx,yy,zz,xy,yz,xz)
static std::vector<double> strainData;   // 6*nElems

// Gauss 2x2x2
static const double GP = 0.5773502691896258; // 1/sqrt(3)
static const double gpts[8][3] = {
    {-GP,-GP,-GP},{GP,-GP,-GP},{GP,GP,-GP},{-GP,GP,-GP},
    {-GP,-GP, GP},{GP,-GP, GP},{GP,GP, GP},{-GP,GP, GP}
};

static void shapeDeriv(double xi, double et, double ze, double dN[8][3]) {
    double xm=1-xi,xp=1+xi,em=1-et,ep=1+et,zm=1-ze,zp=1+ze;
    dN[0][0]=-em*zm; dN[1][0]= em*zm; dN[2][0]= ep*zm; dN[3][0]=-ep*zm;
    dN[4][0]=-em*zp; dN[5][0]= em*zp; dN[6][0]= ep*zp; dN[7][0]=-ep*zp;
    dN[0][1]=-xm*zm; dN[1][1]=-xp*zm; dN[2][1]= xp*zm; dN[3][1]= xm*zm;
    dN[4][1]=-xm*zp; dN[5][1]=-xp*zp; dN[6][1]= xp*zp; dN[7][1]= xm*zp;
    dN[0][2]=-xm*em; dN[1][2]=-xp*em; dN[2][2]=-xp*ep; dN[3][2]=-xm*ep;
    dN[4][2]= xm*em; dN[5][2]= xp*em; dN[6][2]= xp*ep; dN[7][2]= xm*ep;
    for(int i=0;i<8;i++) for(int j=0;j<3;j++) dN[i][j]*=0.125;
}

static double inv3(const double J[3][3], double Ji[3][3]) {
    double d = J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
             - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
             + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
    double id=1.0/d;
    Ji[0][0]=(J[1][1]*J[2][2]-J[1][2]*J[2][1])*id;
    Ji[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])*id;
    Ji[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])*id;
    Ji[1][0]=(J[1][2]*J[2][0]-J[1][0]*J[2][2])*id;
    Ji[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])*id;
    Ji[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])*id;
    Ji[2][0]=(J[1][0]*J[2][1]-J[1][1]*J[2][0])*id;
    Ji[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])*id;
    Ji[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])*id;
    return d;
}

// Find CSR index for (row, col)
static int csrFind(int row, int col) {
    int lo = csrRowPtr[row], hi = csrRowPtr[row+1];
    for (int k = lo; k < hi; k++)
        if (csrColIdx[k] == col) return k;
    return -1;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void generateMesh(double Lx, double Ly, double Lz, int nx, int ny, int nz) {
    g_nx = nx; g_ny = ny; g_nz = nz;
    int nnx=nx+1, nny=ny+1, nnzz=nz+1;
    nNodes = nnx*nny*nnzz;
    nElems = nx*ny*nz;
    nDof = 3*nNodes;
    nodeCoords.resize(3*nNodes);
    elemConn.resize(8*nElems);

    for(int k=0;k<nnzz;k++)
    for(int j=0;j<nny;j++)
    for(int i=0;i<nnx;i++){
        int id=k*nny*nnx+j*nnx+i;
        nodeCoords[3*id]=Lx*i/nx;
        nodeCoords[3*id+1]=Ly*j/ny;
        nodeCoords[3*id+2]=Lz*k/nz;
    }
    int e=0;
    for(int k=0;k<nz;k++)
    for(int j=0;j<ny;j++)
    for(int i=0;i<nx;i++){
        int n0=k*nny*nnx+j*nnx+i;
        elemConn[8*e+0]=n0;
        elemConn[8*e+1]=n0+1;
        elemConn[8*e+2]=n0+nnx+1;
        elemConn[8*e+3]=n0+nnx;
        elemConn[8*e+4]=n0+nny*nnx;
        elemConn[8*e+5]=n0+nny*nnx+1;
        elemConn[8*e+6]=n0+nny*nnx+nnx+1;
        elemConn[8*e+7]=n0+nny*nnx+nnx;
        e++;
    }
}

EMSCRIPTEN_KEEPALIVE
void assembleSystem(double E, double nu, double pressure, int fixFace, int loadFace) {
    // fixFace: 0=X0, 1=XL, 2=Y0, 3=YL, 4=Z0, 5=ZL
    // loadFace: 0=X0, 1=XL, 2=Y0, 3=YL, 4=Z0, 5=ZL
    // D matrix (6x6)
    double lam=E*nu/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
    double D[6][6]; memset(D,0,sizeof(D));
    D[0][0]=D[1][1]=D[2][2]=lam+2*mu;
    D[0][1]=D[0][2]=D[1][0]=D[1][2]=D[2][0]=D[2][1]=lam;
    D[3][3]=D[4][4]=D[5][5]=mu;

    // Build sparsity
    std::vector<std::vector<int>> adj(nDof);
    for(int e=0;e<nElems;e++){
        for(int a=0;a<8;a++)
        for(int b=0;b<8;b++){
            int na=elemConn[8*e+a],nb=elemConn[8*e+b];
            for(int di=0;di<3;di++)
            for(int dj=0;dj<3;dj++)
                adj[3*na+di].push_back(3*nb+dj);
        }
    }
    for(int i=0;i<nDof;i++){
        std::sort(adj[i].begin(),adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(),adj[i].end()),adj[i].end());
    }
    csrRowPtr.resize(nDof+1); csrRowPtr[0]=0;
    for(int i=0;i<nDof;i++) csrRowPtr[i+1]=csrRowPtr[i]+(int)adj[i].size();
    nnz=csrRowPtr[nDof];
    csrColIdx.resize(nnz); csrValues.assign(nnz,0.0);
    for(int i=0;i<nDof;i++)
        for(int k=0;k<(int)adj[i].size();k++)
            csrColIdx[csrRowPtr[i]+k]=adj[i][k];
    adj.clear();

    // Assemble Ke into global CSR
    for(int e=0;e<nElems;e++){
        double Ke[24][24]; memset(Ke,0,sizeof(Ke));
        int nd[8]; for(int i=0;i<8;i++) nd[i]=elemConn[8*e+i];

        for(int g=0;g<8;g++){
            double dNdxi[8][3];
            shapeDeriv(gpts[g][0],gpts[g][1],gpts[g][2],dNdxi);
            double J[3][3]; memset(J,0,sizeof(J));
            for(int a=0;a<8;a++)
            for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                J[i][j]+=dNdxi[a][j]*nodeCoords[3*nd[a]+i];
            double Ji[3][3]; double detJ=inv3(J,Ji);
            double dNdx[8][3];
            for(int a=0;a<8;a++)
            for(int i=0;i<3;i++){
                dNdx[a][i]=0;
                for(int j=0;j<3;j++) dNdx[a][i]+=Ji[i][j]*dNdxi[a][j];
            }
            // B^T D B
            for(int a=0;a<8;a++)
            for(int b=0;b<8;b++){
                // Build B columns for node a and b
                // B_a = [[dNa/dx, 0, 0], [0, dNa/dy, 0], [0, 0, dNa/dz],
                //        [dNa/dy, dNa/dx, 0], [0, dNa/dz, dNa/dy], [dNa/dz, 0, dNa/dx]]
                double Ba[6][3]={}, Bb[6][3]={};
                Ba[0][0]=dNdx[a][0]; Ba[1][1]=dNdx[a][1]; Ba[2][2]=dNdx[a][2];
                Ba[3][0]=dNdx[a][1]; Ba[3][1]=dNdx[a][0];
                Ba[4][1]=dNdx[a][2]; Ba[4][2]=dNdx[a][1];
                Ba[5][0]=dNdx[a][2]; Ba[5][2]=dNdx[a][0];
                Bb[0][0]=dNdx[b][0]; Bb[1][1]=dNdx[b][1]; Bb[2][2]=dNdx[b][2];
                Bb[3][0]=dNdx[b][1]; Bb[3][1]=dNdx[b][0];
                Bb[4][1]=dNdx[b][2]; Bb[4][2]=dNdx[b][1];
                Bb[5][0]=dNdx[b][2]; Bb[5][2]=dNdx[b][0];
                for(int ii=0;ii<3;ii++)
                for(int jj=0;jj<3;jj++){
                    double v=0;
                    for(int p=0;p<6;p++)
                    for(int q=0;q<6;q++)
                        v+=Ba[p][ii]*D[p][q]*Bb[q][jj];
                    Ke[3*a+ii][3*b+jj]+=v*detJ;
                }
            }
        }
        // Scatter into CSR
        for(int a=0;a<8;a++)
        for(int b=0;b<8;b++)
        for(int ii=0;ii<3;ii++)
        for(int jj=0;jj<3;jj++){
            int row=3*nd[a]+ii, col=3*nd[b]+jj;
            int idx=csrFind(row,col);
            if(idx>=0) csrValues[idx]+=Ke[3*a+ii][3*b+jj];
        }
    }

    // RHS: pressure on loadFace
    rhsVec.assign(nDof, 0.0);
    solVec.assign(nDof, 0.0);

    // Compute max coords
    double Lx_val=0, Ly_val=0, Lz_val=0;
    for(int i=0;i<nNodes;i++){
        if(nodeCoords[3*i]>Lx_val) Lx_val=nodeCoords[3*i];
        if(nodeCoords[3*i+1]>Ly_val) Ly_val=nodeCoords[3*i+1];
        if(nodeCoords[3*i+2]>Lz_val) Lz_val=nodeCoords[3*i+2];
    }

    // Determine load face area and normal direction
    // loadFace: 0=X0, 1=XL, 2=Y0, 3=YL, 4=Z0, 5=ZL
    double faceArea = 0;
    int loadDir = 0; // direction of force: 0=x, 1=y, 2=z
    double loadCoord = 0; // coordinate value to match
    int loadAxis = 0; // which axis the face is on
    double sign = 1.0;

    // Pressure (P>0) acts along -outward_normal (compressive, pushing into the body)
    switch(loadFace) {
        case 0: faceArea=Ly_val*Lz_val; loadDir=0; loadAxis=0; loadCoord=0;       sign= 1; break; // X=0: outward -X, force +X
        case 1: faceArea=Ly_val*Lz_val; loadDir=0; loadAxis=0; loadCoord=Lx_val;  sign=-1; break; // X=L: outward +X, force -X
        case 2: faceArea=Lx_val*Lz_val; loadDir=1; loadAxis=1; loadCoord=0;       sign= 1; break; // Y=0: outward -Y, force +Y
        case 3: faceArea=Lx_val*Lz_val; loadDir=1; loadAxis=1; loadCoord=Ly_val;  sign=-1; break; // Y=W: outward +Y, force -Y
        case 4: faceArea=Lx_val*Ly_val; loadDir=2; loadAxis=2; loadCoord=0;       sign= 1; break; // Z=0: outward -Z, force +Z
        case 5: faceArea=Lx_val*Ly_val; loadDir=2; loadAxis=2; loadCoord=Lz_val;  sign=-1; break; // Z=H: outward +Z, force -Z
    }

    // Count nodes on load face
    double tol = std::max({Lx_val, Ly_val, Lz_val}) * 1e-6;
    int loadFaceNodes = 0;
    for(int i=0;i<nNodes;i++){
        if(std::fabs(nodeCoords[3*i+loadAxis] - loadCoord) < tol) loadFaceNodes++;
    }
    if(loadFaceNodes == 0) loadFaceNodes = 1;
    double forcePerNode = pressure * faceArea * sign / loadFaceNodes;
    for(int i=0;i<nNodes;i++){
        if(std::fabs(nodeCoords[3*i+loadAxis] - loadCoord) < tol){
            rhsVec[3*i+loadDir] += forcePerNode;
        }
    }

    // Penalty method for fixFace
    // fixFace: 0=X0, 1=XL, 2=Y0, 3=YL, 4=Z0, 5=ZL
    int fixAxis = 0;
    double fixCoord = 0;
    switch(fixFace) {
        case 0: fixAxis=0; fixCoord=0;       break;
        case 1: fixAxis=0; fixCoord=Lx_val;  break;
        case 2: fixAxis=1; fixCoord=0;       break;
        case 3: fixAxis=1; fixCoord=Ly_val;  break;
        case 4: fixAxis=2; fixCoord=0;       break;
        case 5: fixAxis=2; fixCoord=Lz_val;  break;
    }

    double maxDiag = 0;
    for(int i=0;i<nDof;i++){
        int idx = csrFind(i,i);
        if(idx>=0 && csrValues[idx]>maxDiag) maxDiag=csrValues[idx];
    }
    double penalty = maxDiag * 1e7;
    for(int i=0;i<nNodes;i++){
        if(std::fabs(nodeCoords[3*i+fixAxis] - fixCoord) < tol){
            for(int d=0;d<3;d++){
                int dof = 3*i+d;
                int idx = csrFind(dof,dof);
                if(idx>=0) csrValues[idx] += penalty;
                rhsVec[dof] = 0.0;
            }
        }
    }
}

// Post-processing: compute stress and strain at element centroids
EMSCRIPTEN_KEEPALIVE
void computeResults(double E, double nu) {
    double lam=E*nu/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
    double D[6][6]; memset(D,0,sizeof(D));
    D[0][0]=D[1][1]=D[2][2]=lam+2*mu;
    D[0][1]=D[0][2]=D[1][0]=D[1][2]=D[2][0]=D[2][1]=lam;
    D[3][3]=D[4][4]=D[5][5]=mu;

    stressData.resize(6*nElems);
    strainData.resize(6*nElems);

    for(int e=0;e<nElems;e++){
        int nd[8]; for(int i=0;i<8;i++) nd[i]=elemConn[8*e+i];
        // Evaluate at centroid (0,0,0)
        double dNdxi[8][3];
        shapeDeriv(0,0,0,dNdxi);
        double J[3][3]; memset(J,0,sizeof(J));
        for(int a=0;a<8;a++)
        for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            J[i][j]+=dNdxi[a][j]*nodeCoords[3*nd[a]+i];
        double Ji[3][3]; inv3(J,Ji);
        double dNdx[8][3];
        for(int a=0;a<8;a++)
        for(int i=0;i<3;i++){
            dNdx[a][i]=0;
            for(int j=0;j<3;j++) dNdx[a][i]+=Ji[i][j]*dNdxi[a][j];
        }
        // strain = B * u_e
        double eps[6]={};
        for(int a=0;a<8;a++){
            double ux=solVec[3*nd[a]], uy=solVec[3*nd[a]+1], uz=solVec[3*nd[a]+2];
            eps[0]+=dNdx[a][0]*ux;
            eps[1]+=dNdx[a][1]*uy;
            eps[2]+=dNdx[a][2]*uz;
            eps[3]+=dNdx[a][1]*ux+dNdx[a][0]*uy;
            eps[4]+=dNdx[a][2]*uy+dNdx[a][1]*uz;
            eps[5]+=dNdx[a][2]*ux+dNdx[a][0]*uz;
        }
        for(int i=0;i<6;i++) strainData[6*e+i]=eps[i];
        // stress = D * strain
        for(int i=0;i<6;i++){
            double s=0;
            for(int j=0;j<6;j++) s+=D[i][j]*eps[j];
            stressData[6*e+i]=s;
        }
    }
}

// Accessor functions for JS
EMSCRIPTEN_KEEPALIVE int getNumNodes() { return nNodes; }
EMSCRIPTEN_KEEPALIVE int getNumElems() { return nElems; }
EMSCRIPTEN_KEEPALIVE int getNumDof()   { return nDof; }
EMSCRIPTEN_KEEPALIVE int getNnz()      { return nnz; }
EMSCRIPTEN_KEEPALIVE double* getNodeCoords() { return nodeCoords.data(); }
EMSCRIPTEN_KEEPALIVE int*    getElemConn()   { return elemConn.data(); }
EMSCRIPTEN_KEEPALIVE int*    getCsrRowPtr()  { return csrRowPtr.data(); }
EMSCRIPTEN_KEEPALIVE int*    getCsrColIdx()  { return csrColIdx.data(); }
EMSCRIPTEN_KEEPALIVE double* getCsrValues()  { return csrValues.data(); }
EMSCRIPTEN_KEEPALIVE double* getRhsVec()     { return rhsVec.data(); }
EMSCRIPTEN_KEEPALIVE double* getSolVec()     { return solVec.data(); }
EMSCRIPTEN_KEEPALIVE double* getStressData() { return stressData.data(); }
EMSCRIPTEN_KEEPALIVE double* getStrainData() { return strainData.data(); }

// Allow JS to write solution back
EMSCRIPTEN_KEEPALIVE void setSolution(int i, double v) { solVec[i] = v; }

} // extern "C"