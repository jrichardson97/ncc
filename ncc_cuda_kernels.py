CORRELATE_STANDARD_SRC = '''
extern "C"

__global__ void correlate(
        float *c, float *rf, 
        int k, 
        int nLat, int nT, int nAx, 
        int nAxTrack, int nShifts,
        int axSparsity, int maximizeThreadsPerBlock
    ) 
{
    int iLat = blockIdx.y;
    int iAxTrack = blockIdx.x;
    int iShift = threadIdx.x;

    int iT = 0;
    if (maximizeThreadsPerBlock > 0) {
        iT = threadIdx.y;
    } else {
        iT = blockIdx.z;
    }

    int tauMax = (nShifts - 1) / 2;

    // ind(x,y,z) = z + y*nz + x*nz*ny
    // z = (nLat, nT, nAx) = (x, y, z)
    int iAx = iAxTrack * axSparsity;    // z index 
    int iRef = iAx + iLat*nAx*nT;       // ref line when iT = 0, iT = y so middle term = 0
    int iTrack = iRef + (iT+1)*nAx;      // track lines, add y term to 

    // ind(w,x,y,z) = z + y*nz + x*nz*ny + w*nz*ny*nx
    // c = (nLat, nAxTrack, nT-1, nShifts) = (w, x, y, z)
    int iCC = iShift + iT*nShifts + iAxTrack*nShifts*(nT-1) + iLat*nShifts*(nT-1)*nAxTrack;

    // Get mean of x and y
    float xbar = 0.f;
    float ybar = 0.f;
    for (int ik = 0; ik < k; ik++) 
    {
        // Load data
        float xk = rf[ik + iRef + tauMax];
        float yk = rf[ik + iTrack + iShift];

        xbar += xk;
        ybar += yk;
    }

    xbar = xbar / k;
    ybar = ybar / k;

    // Use a running sum
    float tmp1 = 0.f; // sum((x-xbar)*(y-ybar))
    float tmp2 = 0.f; // sum((x-xbar)^2)
    float tmp3 = 0.f; // sum((y-ybar)^2)
    for (int ik = 0; ik < k; ik++) 
    {
        // Load data
        float xk = rf[ik + iRef + tauMax];
        float yk = rf[ik + iTrack + iShift];

        float xcenter = xk-xbar;
        float ycenter = yk-ybar;
        tmp1 += xcenter*ycenter;
        tmp2 += xcenter*xcenter;
        tmp3 += ycenter*ycenter;
    }

    c[iCC] = tmp1 / (sqrt(tmp2) * sqrt(tmp3));
}
'''

CORRELATE_SINGLEPASS_SRC = '''
extern "C"

__global__ void correlate(
        float *c, float *rf, 
        int k, 
        int nLat, int nT, int nAx, 
        int nAxTrack, int nShifts,
        int axSparsity, int maximizeThreadsPerBlock
    ) 
{
    int iLat = blockIdx.y;
    int iAxTrack = blockIdx.x;
    int iShift = threadIdx.x;

    int iT = 0;
    if (maximizeThreadsPerBlock > 0) {
        iT = threadIdx.y;
    } else {
        iT = blockIdx.z;
    }

    int tauMax = (nShifts - 1) / 2;

    // ind(x,y,z) = z + y*nz + x*nz*ny
    // z = (nLat, nT, nAx) = (x, y, z)
    int iAx = iAxTrack * axSparsity;    // z index 
    int iRef = iAx + iLat*nAx*nT;       // ref line when iT = 0, iT = y so middle term = 0
    int iTrack = iRef + (iT+1)*nAx;      // track lines, add y term to 

    // ind(w,x,y,z) = z + y*nz + x*nz*ny + w*nz*ny*nx
    // c = (nLat, nAxTrack, nT-1, nShifts) = (w, x, y, z)
    int iCC = iShift + iT*nShifts + iAxTrack*nShifts*(nT-1) + iLat*nShifts*(nT-1)*nAxTrack;

    // Use a running sum
    float xxsum = 0.f; // sum(x^2)
    float yysum = 0.f; // sum(y^2)
    float xysum = 0.f; // sum(x*y)
    float xsum = 0.f;  // sum(x)
    float ysum = 0.f;  // sum(y)

    // Loop over axial kernel
    for (int ik = 0; ik < k; ik++) 
    {
        // Load data
        float xk = rf[ik + iRef + tauMax];
        float yk = rf[ik + iTrack + iShift];

        // Compute correlation components
        xxsum += xk * xk;
        yysum += yk * yk;
        xysum += xk * yk;
        xsum += xk;
        ysum += yk;
    }

    float tmp1 = k * xysum - xsum * ysum; // E[xy] - E[x]E[y]
    float tmp2 = k * xxsum - xsum * xsum; // E[x^2] - E[x]^2
    float tmp3 = k * yysum - ysum * ysum; // E[y^2] - E[y]^2

    c[iCC] = tmp1 / (sqrt(tmp2) * sqrt(tmp3));
}
'''

PARABOLIC_INTERPOLATION_SRC = '''
extern "C"

__global__ void pinterp(
        float *cc, float *arfidata, 
        float *cc_all, int *idxCcMaxArr,
        int nShifts
    ) 
{
    int iRow = blockIdx.x;

    int tauMax = (nShifts - 1) / 2;

    int idxCcMax = idxCcMaxArr[iRow];
    int iShift = iRow*nShifts + idxCcMax;

    float subsampleShift = 0;
    float ccMax = 0;
    if (idxCcMax > 0 && idxCcMax < nShifts - 1) {
        float y1 = cc_all[iShift-1];
        float y2 = cc_all[iShift];
        float y3 = cc_all[iShift+1];

        ccMax = (y2+(-y1*y1+2*y1*y3-y3*y3)/(8*y1-16*y2+8*y3));
        subsampleShift = ((y1-y3)/(2*y1-4*y2+2*y3));
    } else {
        ccMax = cc_all[iShift];
    }

    cc[iRow] = ccMax;
    arfidata[iRow] = idxCcMax - tauMax + subsampleShift;
}
'''
