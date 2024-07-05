from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from .ncc_cuda_kernels import (
    CORRELATE_SINGLEPASS_SRC,
    CORRELATE_STANDARD_SRC,
    PARABOLIC_INTERPOLATION_SRC,
)
from .ncc_utils import nthMoment, parabolicInterpolation, upsample_rf


def corrcoeff_singlepass(x, y):
    k = len(x)

    xysum = x @ y
    xxsum = x @ x
    yysum = y @ y
    ysum = np.sum(y)
    xsum = np.sum(x)

    tmp1 = k * xysum - xsum * ysum  # E[xy] - E[x]E[y]
    tmp2 = k * xxsum - xsum * xsum  # E[x^2] - E[x]^2
    tmp3 = k * yysum - ysum * ysum  # E[y^2] - E[y]^2

    return tmp1 / (np.sqrt(tmp2) * np.sqrt(tmp3))


def corrcoeff(x, y):
    xcenter = x - np.mean(x)
    ycenter = y - np.mean(y)

    tmp1 = xcenter @ ycenter  # E[(x-xbar)(y-ybar)]
    tmp2 = xcenter @ xcenter  # E[(x-xbar)^2]
    tmp3 = ycenter @ ycenter  # E[(y-ybar)^2]

    return tmp1 / np.sqrt(tmp2 * tmp3)


def corrcoeff_moments(x, y):
    k = len(x)

    xMom1 = nthMoment(x, 1, k)
    xMom2 = nthMoment(x, 2, k)
    yMom1 = nthMoment(y, 1, k)
    yMom2 = nthMoment(y, 2, k)

    xVar = xMom2 - xMom1**2
    yVar = yMom2 - yMom1**2

    return (x - xMom1) @ (y - yMom1) / (k * np.sqrt(xVar * yVar))


@dataclass
class NccParams:
    fc: float  # transducer center frequency (Hz)
    c: float = 1540  # speed of sound (m/s)
    fs: float = 40e6  # sampling frequency (Hz)
    fsUpsampleFactor: int = 4  # rf axial interpolation upsampling factor
    axSparsity: int = 1  #
    nLambdaKernel: float = 2  # cross correlation kernel size (n wavelengths)
    maxShift: float = 10e-6  # maximum expected shift in each direction (microns)
    lamb: float = field(init=False)  # pulse wavelength (m)

    def __post_init__(self):
        self.lamb = self.c / self.fc


def ncc(
    rf,
    fc: float,
    c: float = 1540,
    fs: float = 40e6,
    fsUpsampleFactor: int = 4,
    axSparsity: int = 1,
    nLambdaKernel: float = 2,
    maxShift: float = 10e-6,
    lamb: float = None,
    mode="standard-fast-gpu",
):
    if lamb is None:
        lamb = c / fc

    if rf.ndim == 2:
        rf = rf[:, np.newaxis, :]

    if fsUpsampleFactor > 1:
        rf = upsample_rf(rf, fsUpsampleFactor)

    nccKernelLengthSamples = round(nLambdaKernel * lamb * 2 / c * fs)
    metersPerSample = c / (2 * fs)
    maxShiftSamples = np.ceil(maxShift / metersPerSample)

    tauMax = int(fsUpsampleFactor * maxShiftSamples)
    metersPerSample = metersPerSample / fsUpsampleFactor

    k = fsUpsampleFactor * nccKernelLengthSamples
    nshifts = 2 * tauMax + 1
    sparsity = fsUpsampleFactor * axSparsity

    if "cpu" in mode:
        cc, arfidata = _ncc_cpu(rf, k, nshifts, sparsity, mode=mode)
    elif "gpu" in mode:
        cc, arfidata = _ncc_gpu(rf, k, nshifts, sparsity, mode=mode)
    else:
        raise ValueError(f"Mode must contain either 'cpu' or 'gpu'. Mode: {mode}")

    # Convert arfidata from fractional sample shift to microns
    micronsPerMeter = 1e6
    arfidata = arfidata * metersPerSample * micronsPerMeter

    return cc, arfidata


def _ncc_gpu(rf, k, nshifts, axSparsity, mode):
    nAx0, nLat0, nT0 = rf.shape
    rf = np.transpose(rf, (1, 2, 0))
    nLat, nT, nAx = rf.shape
    assert nLat == nLat0
    assert nAx == nAx0
    assert nT == nT0

    tauMax = int((nshifts - 1) / 2)
    axSliceIndexArr = np.arange(0, (nAx - k - tauMax), axSparsity)
    nAxTrack = len(axSliceIndexArr)

    # Move rf to GPU.
    rf_gpu = cp.asarray(rf)

    # Create output array in the GPU memory.
    cc_all_gpu = cp.zeros(shape=(nLat, nAxTrack, nT - 1, nshifts), dtype=rf.dtype)

    # Determine CUDA kernel grid dimensions.
    if "fast" in mode:
        block_size = (nshifts, nT - 1)
        grid_size = (nAxTrack, nLat)
        maximizeThreadsPerBlock = 1
    else:
        block_size = (nshifts,)
        grid_size = (nAxTrack, nLat, nT - 1)
        maximizeThreadsPerBlock = 0

    if "singlepass" in mode:
        ncc_kernel = cp.RawKernel(code=CORRELATE_SINGLEPASS_SRC, name="correlate")
    elif "standard" in mode:
        ncc_kernel = cp.RawKernel(code=CORRELATE_STANDARD_SRC, name="correlate")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    print("NCC GPU Allocation:")
    print(f"\tBlock size : {block_size}")
    print(f"\tGrid size  : {grid_size}")
    print(f"Total correlation calls = {nshifts*nAxTrack*nLat*(nT-1)}")

    # Call ncc cuda kernel
    ncc_kernel_params = (
        cc_all_gpu,
        rf_gpu,
        k,
        nLat,
        nT,
        nAx,
        nAxTrack,
        nshifts,
        axSparsity,
        maximizeThreadsPerBlock,
    )
    ncc_kernel(grid_size, block_size, ncc_kernel_params)

    # Find index of max correlation along tau shift axis
    cc_all_gpu = cc_all_gpu.reshape(-1, nshifts)
    idxCcMaxArr_gpu = cp.argmax(cc_all_gpu, axis=1, dtype=cp.int32)

    # Initialize final correlation coefficient and arfidata matrices on the gpu
    n = nLat * nAxTrack * (nT - 1)
    cc_gpu = cp.zeros(shape=(n,), dtype=rf.dtype)
    arfidata_gpu = cp.zeros(shape=(n,), dtype=rf.dtype)

    # Parabolic interpolation of correlation coeffs to get subsample cc and displacements
    pinterp_kernel = cp.RawKernel(code=PARABOLIC_INTERPOLATION_SRC, name="pinterp")
    block_size = (1,)
    grid_size = (n,)
    print("Pinterp GPU Allocation:")
    print(f"\tBlock size : {block_size}")
    print(f"\tGrid size  : {grid_size}")
    print(f"Total pinterp calls = {n}")
    pinterp_kernel_params = (cc_gpu, arfidata_gpu, cc_all_gpu, idxCcMaxArr_gpu, nshifts)
    pinterp_kernel(grid_size, block_size, pinterp_kernel_params)

    # Wait for computation finish, transfer to cpu, and reshape then transpose
    cc = cc_gpu.get().reshape(nLat, nAxTrack, nT - 1)
    arfidata = arfidata_gpu.get().reshape(nLat, nAxTrack, nT - 1)
    cc = np.transpose(cc, (1, 0, 2))
    arfidata = np.transpose(arfidata, (1, 0, 2))

    return cc, arfidata


def _ncc_cpu(rf, k, nshifts, axSparsity, mode):
    nAx0, nLat0, nT0 = rf.shape
    rf = np.transpose(rf, (1, 2, 0))
    nLat, nT, nAx = rf.shape
    assert nLat == nLat0
    assert nAx == nAx0
    assert nT == nT0

    if "singlepass" in mode:
        corr_func = corrcoeff_singlepass
    elif "standard" in mode:
        corr_func = corrcoeff
    elif "moments" in mode:
        corr_func = corrcoeff_moments
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Uncomment for linear indexing
    rf1 = rf.reshape(
        -1,
    )

    tauMax = int((nshifts - 1) / 2)
    nAxSlice = k + nshifts

    axSliceIndexArr = np.arange(0, (nAx - k - tauMax), axSparsity)
    nAxTrack = len(axSliceIndexArr)

    cc_tmp = np.empty((nLat, nAxTrack, nT - 1, nshifts), dtype=np.float32)

    for iLat in range(nLat):
        for iAxTrack in range(nAxTrack):
            for iT in range(nT - 1):
                for iShift in range(nshifts):
                    # z + y*nz + x*nz*ny
                    # Uncomment for linear indexing
                    iAx = iAxTrack * axSparsity
                    zix = iAx + 0 * nAx + iLat * nAx * nT
                    ziy = zix + (iT + 1) * nAx

                    x = rf1[zix + tauMax : zix + nAxSlice - tauMax - 1]
                    y = rf1[ziy + iShift : ziy + nAxSlice - (nshifts - iShift)]

                    # Uncomment for multidimensional indexing
                    # axSliceIndex = axSliceIndexArr[iAxTrack]
                    # x = rf[iLat,0,axSliceIndex+tauMax:axSliceIndex+nAxSlice-tauMax-1]
                    # y = rf[iLat,iT+1,axSliceIndex+iShift:axSliceIndex+nAxSlice-(nshifts-iShift)]

                    cc_tmp[iLat, iAxTrack, iT, iShift] = corr_func(x, y)

    nLat, nAx, nT, nShifts = cc_tmp.shape

    cc = np.zeros(shape=(nLat * nAx * nT,), dtype=np.float32)
    arfidata = np.zeros(shape=(nLat * nAx * nT,), dtype=np.float32)

    cc_tmp = cc_tmp.reshape(-1, nShifts)
    idxCcMaxArr = np.argmax(cc_tmp, axis=1)

    for ii in range(nAx * nLat * nT):
        idxCcMax = idxCcMaxArr[ii]
        ccMaxSampleShift = idxCcMax - tauMax

        cc_shifts = cc_tmp[ii, :]

        if (idxCcMax > 0) and (idxCcMax < nshifts - 1):
            [ccMax, subsampleShift] = parabolicInterpolation(
                cc_shifts[idxCcMax - 1], cc_shifts[idxCcMax], cc_shifts[idxCcMax + 1]
            )
        else:
            subsampleShift = 0
            ccMax = cc_shifts[idxCcMax]

        cc[ii] = ccMax
        arfidata[ii] = ccMaxSampleShift + subsampleShift

    cc = cc.reshape(nLat, nAx, nT)
    cc = np.transpose(cc, (1, 0, 2))
    arfidata = arfidata.reshape(nLat, nAx, nT)
    arfidata = np.transpose(arfidata, (1, 0, 2))

    return cc, arfidata
