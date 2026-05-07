namespace ChebyshevSharp.Internal;

/// <summary>
/// Sobol sensitivity indices computed from Chebyshev spectral coefficients.
/// Mirrors PyChebyshev <c>_sensitivity.py</c> (v0.20.0).
/// </summary>
internal static class Sensitivity
{
    // DCT and TT contractions can leave tiny nonzero energy for exactly constant
    // functions. Use a scale-relative floor so valid low-amplitude signals are not
    // erased by a fixed absolute cutoff.
    private const double RelativeVarianceNoiseFloor = 1e-28;

    /// <summary>Chebyshev T_n inner product norm² under weight 1/√(1-x²) on [-1,1].</summary>
    private static double ChebyshevNormSquared(int n) => n == 0 ? Math.PI : Math.PI / 2.0;

    /// <summary>Multi-D inner product norm² = ∏ per-dim norms.</summary>
    private static double MultiIndexNormSquared(int[] alpha)
    {
        double r = 1.0;
        for (int i = 0; i < alpha.Length; i++) r *= ChebyshevNormSquared(alpha[i]);
        return r;
    }

    private static bool IsNumericalZeroVariance(double variance, double totalWeightedSquared)
    {
        if (variance <= 0.0) return true;
        double scale = Math.Abs(totalWeightedSquared);
        return variance <= scale * RelativeVarianceNoiseFloor;
    }

    private static void AddWeightedCoreContraction(
        double[,] source,
        TensorTrainKernel.TtCore core,
        double[] weights,
        int startDegree,
        int endDegreeExclusive,
        double[,] target)
    {
        int rL = core.RLeft, rR = core.RRight;
        for (int a = 0; a < rR; a++)
            for (int b = 0; b < rR; b++)
            {
                double acc = 0.0;
                for (int i = 0; i < rL; i++)
                    for (int j = 0; j < rL; j++)
                    {
                        double sij = source[i, j];
                        if (sij == 0.0) continue;
                        for (int p = startDegree; p < endDegreeExclusive; p++)
                            acc += sij * core[i, p, a] * weights[p] * core[j, p, b];
                    }
                target[a, b] += acc;
            }
    }

    private static double ContractNonconstantWeightedSquared(
        TensorTrainKernel.TtCore[] cores,
        double[][] weights)
    {
        var zero = new double[1, 1];
        var nonzero = new double[1, 1];
        zero[0, 0] = 1.0;

        for (int k = 0; k < cores.Length; k++)
        {
            var core = cores[k];
            var nextZero = new double[core.RRight, core.RRight];
            var nextNonzero = new double[core.RRight, core.RRight];

            // Still all-zero multi-index: only degree 0 may be selected.
            AddWeightedCoreContraction(zero, core, weights[k], 0, 1, nextZero);

            // Already nonzero: any degree keeps the path nonconstant.
            AddWeightedCoreContraction(nonzero, core, weights[k], 0, core.NNodes, nextNonzero);

            // First nonzero degree can occur in this core.
            AddWeightedCoreContraction(zero, core, weights[k], 1, core.NNodes, nextNonzero);

            zero = nextZero;
            nonzero = nextNonzero;
        }

        return nonzero[0, 0];
    }

    /// <summary>Convert flat row-major index → multi-index (one int per dimension).</summary>
    private static int[] UnravelIndex(long flat, int[] shape)
    {
        int n = shape.Length;
        var idx = new int[n];
        long rem = flat;
        for (int d = n - 1; d >= 0; d--)
        {
            idx[d] = (int)(rem % shape[d]);
            rem /= shape[d];
        }
        return idx;
    }

    private static long RavelIndex(int[] idx, int[] shape)
    {
        long flat = 0;
        for (int d = 0; d < shape.Length; d++)
            flat = checked(flat * shape[d] + idx[d]);
        return flat;
    }

    private static int[] RemoveDimension(int[] values, int dim)
    {
        var result = new int[values.Length - 1];
        for (int src = 0, dst = 0; src < values.Length; src++)
            if (src != dim) result[dst++] = values[src];
        return result;
    }

    /// <summary>
    /// Apply <see cref="BarycentricKernel.ChebyshevCoefficients1D"/> along every axis,
    /// matching PyChebyshev's <c>_compute_chebyshev_coefficients</c> convention
    /// (DCT-II per axis with c_0 halving, applied dim-by-dim).
    /// </summary>
    /// <param name="tensorValues">Row-major tensor of values at Type-I Chebyshev nodes.</param>
    /// <param name="shape">Per-dim node counts.</param>
    /// <returns>Row-major tensor of Chebyshev coefficients, same shape.</returns>
    internal static double[] ChebyshevCoefficientsND(double[] tensorValues, int[] shape)
    {
        int nDim = shape.Length;
        var coeffs = (double[])tensorValues.Clone();

        // Apply 1D DCT-II axis-by-axis.
        // For each axis d, compute the leading "outer" size (product of dims to the
        // left of d) and the trailing "inner" size (product of dims to the right).
        // Iterate over (outer, inner) coordinates, extract a 1D slice along d,
        // apply BarycentricKernel.ChebyshevCoefficients1D, write back.
        for (int d = 0; d < nDim; d++)
        {
            int n = shape[d];
            int outer = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(shape.Take(d), nameof(ChebyshevCoefficientsND)),
                nameof(ChebyshevCoefficientsND));
            int inner = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(shape.Skip(d + 1), nameof(ChebyshevCoefficientsND)),
                nameof(ChebyshevCoefficientsND));

            var slice = new double[n];
            for (int o = 0; o < outer; o++)
            {
                for (int j = 0; j < inner; j++)
                {
                    // Extract slice: coeffs[o, :, j] in (outer, n, inner) layout.
                    for (int i = 0; i < n; i++)
                        slice[i] = coeffs[(o * n + i) * inner + j];

                    var c = BarycentricKernel.ChebyshevCoefficients1D(slice);
                    // c[0] is already halved by ChebyshevCoefficients1D (matches PyChebyshev convention).

                    for (int i = 0; i < n; i++)
                        coeffs[(o * n + i) * inner + j] = c[i];
                }
            }
        }

        return coeffs;
    }

    /// <summary>
    /// Compute Sobol indices for piecewise Chebyshev coefficients on a tensor-product
    /// interval partition. This includes both local polynomial variance within each
    /// piece and variance from different piece means.
    /// </summary>
    internal static SobolResult ComputeSobolFromPiecewiseCoeffs(
        double[][] pieceCoeffs,
        int[][] pieceCoeffShapes,
        int[] pieceShape,
        double[][] intervalLengths)
    {
        int nDim = pieceShape.Length;
        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(pieceShape, nameof(ComputeSobolFromPiecewiseCoeffs)),
            nameof(ComputeSobolFromPiecewiseCoeffs),
            pieceShape);

        if (pieceCoeffs.Length != totalPieces)
            throw new ArgumentException(
                $"pieceCoeffs.Length={pieceCoeffs.Length} does not match prod(pieceShape)={totalPieces}.");
        if (pieceCoeffShapes.Length != totalPieces)
            throw new ArgumentException(
                $"pieceCoeffShapes.Length={pieceCoeffShapes.Length} does not match prod(pieceShape)={totalPieces}.");
        if (intervalLengths.Length != nDim)
            throw new ArgumentException(
                $"intervalLengths.Length={intervalLengths.Length} != pieceShape.Length={nDim}.");

        var totalLengths = new double[nDim];
        for (int d = 0; d < nDim; d++)
        {
            if (intervalLengths[d].Length != pieceShape[d])
                throw new ArgumentException(
                    $"intervalLengths[{d}].Length={intervalLengths[d].Length} != pieceShape[{d}]={pieceShape[d]}.");
            for (int i = 0; i < intervalLengths[d].Length; i++)
            {
                double length = intervalLengths[d][i];
                if (!double.IsFinite(length) || length <= 0.0)
                    throw new ArgumentException("interval lengths must be finite and positive.");
                totalLengths[d] += length;
            }
        }

        var pieceIndices = new int[totalPieces][];
        double totalVolume = 1.0;
        double totalMeasure = 1.0;
        for (int d = 0; d < nDim; d++)
        {
            totalVolume *= totalLengths[d];
            totalMeasure *= totalLengths[d] * Math.PI;
        }

        double weightedMeanSum = 0.0;
        double totalWeightedSquared = 0.0;
        for (int flatPiece = 0; flatPiece < totalPieces; flatPiece++)
        {
            int[] pieceIndex = UnravelIndex(flatPiece, pieceShape);
            pieceIndices[flatPiece] = pieceIndex;

            double volume = 1.0;
            for (int d = 0; d < nDim; d++)
                volume *= intervalLengths[d][pieceIndex[d]];

            int[] coeffShape = pieceCoeffShapes[flatPiece];
            double[] coeffs = pieceCoeffs[flatPiece];
            int expectedCoeffLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(coeffShape, nameof(ComputeSobolFromPiecewiseCoeffs)),
                nameof(ComputeSobolFromPiecewiseCoeffs),
                coeffShape);
            if (coeffs.Length != expectedCoeffLength)
                throw new ArgumentException(
                    $"pieceCoeffs[{flatPiece}].Length={coeffs.Length} does not match prod(pieceCoeffShapes[{flatPiece}])={expectedCoeffLength}.");

            weightedMeanSum += volume * coeffs[0];
            for (long flatCoeff = 0; flatCoeff < coeffs.Length; flatCoeff++)
            {
                double coeff = coeffs[flatCoeff];
                if (!double.IsFinite(coeff))
                    throw new ArgumentException(
                        "coefficients contain NaN or Inf; SobolIndices() requires finite spectral coefficients");

                if (coeff == 0.0) continue;
                int[] alpha = UnravelIndex(flatCoeff, coeffShape);
                totalWeightedSquared += volume * coeff * coeff * MultiIndexNormSquared(alpha);
            }
        }

        double globalMean = weightedMeanSum / totalVolume;
        double variance = totalWeightedSquared - totalMeasure * globalMean * globalMean;
        if (IsNumericalZeroVariance(variance, totalWeightedSquared))
            return new SobolResult(new double[nDim], new double[nDim], variance);

        var firstOrder = new double[nDim];
        var totalOrder = new double[nDim];

        for (int d = 0; d < nDim; d++)
        {
            firstOrder[d] = FirstProjectionEnergy(
                d,
                globalMean,
                pieceCoeffs,
                pieceCoeffShapes,
                pieceShape,
                pieceIndices,
                intervalLengths,
                totalLengths);

            double complementEnergy = ComplementProjectionEnergy(
                d,
                globalMean,
                pieceCoeffs,
                pieceCoeffShapes,
                pieceShape,
                pieceIndices,
                intervalLengths,
                totalLengths);

            double totalEnergy = variance - complementEnergy;
            if (totalEnergy < 0.0 && totalEnergy > -Math.Abs(variance) * 1e-12)
                totalEnergy = 0.0;
            if (totalEnergy > variance && totalEnergy - variance < Math.Abs(variance) * 1e-12)
                totalEnergy = variance;
            totalOrder[d] = totalEnergy;
        }

        for (int d = 0; d < nDim; d++)
        {
            firstOrder[d] /= variance;
            totalOrder[d] /= variance;
        }

        return new SobolResult(firstOrder, totalOrder, variance);
    }

    private static double FirstProjectionEnergy(
        int dim,
        double globalMean,
        double[][] pieceCoeffs,
        int[][] pieceCoeffShapes,
        int[] pieceShape,
        int[][] pieceIndices,
        double[][] intervalLengths,
        double[] totalLengths)
    {
        var projectedCoeffs = new double[pieceShape[dim]][];
        for (int interval = 0; interval < projectedCoeffs.Length; interval++)
            projectedCoeffs[interval] = Array.Empty<double>();

        double measureOther = 1.0;
        double volumeOther = 1.0;
        for (int d = 0; d < pieceShape.Length; d++)
        {
            if (d == dim) continue;
            measureOther *= totalLengths[d] * Math.PI;
            volumeOther *= totalLengths[d];
        }

        for (int flatPiece = 0; flatPiece < pieceCoeffs.Length; flatPiece++)
        {
            int[] pieceIndex = pieceIndices[flatPiece];
            int interval = pieceIndex[dim];
            int[] coeffShape = pieceCoeffShapes[flatPiece];
            double[] coeffs = pieceCoeffs[flatPiece];

            if (projectedCoeffs[interval].Length < coeffShape[dim])
                Array.Resize(ref projectedCoeffs[interval], coeffShape[dim]);

            double weight = 1.0;
            for (int d = 0; d < pieceShape.Length; d++)
            {
                if (d == dim) continue;
                weight *= intervalLengths[d][pieceIndex[d]];
            }
            weight /= volumeOther;

            var alpha = new int[coeffShape.Length];
            for (int mode = 0; mode < coeffShape[dim]; mode++)
            {
                alpha[dim] = mode;
                projectedCoeffs[interval][mode] += weight * coeffs[RavelIndex(alpha, coeffShape)];
            }
        }

        double univariateEnergy = 0.0;
        for (int interval = 0; interval < projectedCoeffs.Length; interval++)
        {
            double length = intervalLengths[dim][interval];
            var coeffs = projectedCoeffs[interval];
            if (coeffs.Length == 0)
                continue;

            double meanDelta = coeffs[0] - globalMean;
            univariateEnergy += length * meanDelta * meanDelta * ChebyshevNormSquared(0);
            for (int mode = 1; mode < coeffs.Length; mode++)
                univariateEnergy += length * coeffs[mode] * coeffs[mode] * ChebyshevNormSquared(mode);
        }

        return measureOther * univariateEnergy;
    }

    private static double ComplementProjectionEnergy(
        int dim,
        double globalMean,
        double[][] pieceCoeffs,
        int[][] pieceCoeffShapes,
        int[] pieceShape,
        int[][] pieceIndices,
        double[][] intervalLengths,
        double[] totalLengths)
    {
        int nDim = pieceShape.Length;
        if (nDim == 1) return 0.0;

        int[] complementPieceShape = RemoveDimension(pieceShape, dim);
        int complementPieceCount = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(complementPieceShape, nameof(ComplementProjectionEnergy)),
            nameof(ComplementProjectionEnergy),
            complementPieceShape);
        var complementCoeffShapes = new int[complementPieceCount][];
        for (int i = 0; i < complementPieceCount; i++)
            complementCoeffShapes[i] = new int[nDim - 1];

        for (int flatPiece = 0; flatPiece < pieceCoeffs.Length; flatPiece++)
        {
            int compFlat = RavelComplementPiece(pieceIndices[flatPiece], dim, complementPieceShape);
            int[] coeffShape = pieceCoeffShapes[flatPiece];
            for (int src = 0, dst = 0; src < nDim; src++)
            {
                if (src == dim) continue;
                complementCoeffShapes[compFlat][dst] = Math.Max(
                    complementCoeffShapes[compFlat][dst],
                    coeffShape[src]);
                dst++;
            }
        }

        var projected = new Dictionary<long, double>[complementPieceCount];
        for (int i = 0; i < projected.Length; i++)
            projected[i] = new Dictionary<long, double>();

        for (int flatPiece = 0; flatPiece < pieceCoeffs.Length; flatPiece++)
        {
            int[] pieceIndex = pieceIndices[flatPiece];
            int compFlat = RavelComplementPiece(pieceIndex, dim, complementPieceShape);
            int[] coeffShape = pieceCoeffShapes[flatPiece];
            double[] coeffs = pieceCoeffs[flatPiece];
            int[] compCoeffShape = complementCoeffShapes[compFlat];
            double weight = intervalLengths[dim][pieceIndex[dim]] / totalLengths[dim];

            for (long flatCoeff = 0; flatCoeff < coeffs.Length; flatCoeff++)
            {
                double coeff = coeffs[flatCoeff];
                if (coeff == 0.0) continue;

                int[] alpha = UnravelIndex(flatCoeff, coeffShape);
                if (alpha[dim] != 0) continue;

                int[] compAlpha = RemoveDimension(alpha, dim);
                long compKey = RavelIndex(compAlpha, compCoeffShape);
                projected[compFlat].TryGetValue(compKey, out double current);
                projected[compFlat][compKey] = current + weight * coeff;
            }
        }

        double complementTotalWeightedSquared = 0.0;
        for (int compFlat = 0; compFlat < complementPieceCount; compFlat++)
        {
            int[] compPieceIndex = UnravelIndex(compFlat, complementPieceShape);
            double compVolume = 1.0;
            for (int src = 0, dst = 0; src < nDim; src++)
            {
                if (src == dim) continue;
                compVolume *= intervalLengths[src][compPieceIndex[dst]];
                dst++;
            }

            int[] compCoeffShape = complementCoeffShapes[compFlat];
            foreach (var kvp in projected[compFlat])
            {
                int[] alpha = UnravelIndex(kvp.Key, compCoeffShape);
                complementTotalWeightedSquared += compVolume * kvp.Value * kvp.Value * MultiIndexNormSquared(alpha);
            }
        }

        double complementMeasure = 1.0;
        for (int d = 0; d < nDim; d++)
            if (d != dim) complementMeasure *= totalLengths[d] * Math.PI;

        double complementVariance = complementTotalWeightedSquared - complementMeasure * globalMean * globalMean;
        if (complementVariance < 0.0 && complementVariance > -Math.Abs(complementTotalWeightedSquared) * 1e-12)
            complementVariance = 0.0;

        return totalLengths[dim] * Math.PI * complementVariance;
    }

    private static int RavelComplementPiece(int[] pieceIndex, int removedDim, int[] complementPieceShape)
    {
        if (complementPieceShape.Length == 0) return 0;

        var compIndex = new int[complementPieceShape.Length];
        for (int src = 0, dst = 0; src < pieceIndex.Length; src++)
            if (src != removedDim) compIndex[dst++] = pieceIndex[src];

        return TensorShape.RequireArrayLength(
            RavelIndex(compIndex, complementPieceShape),
            nameof(RavelComplementPiece),
            complementPieceShape);
    }

    /// <summary>
    /// Compute first-order and total-order Sobol sensitivity indices directly from
    /// TT coefficient cores. O(d · n · r²) per dimension — no dense materialization.
    /// Mirrors PyChebyshev <c>_sensitivity.py:_compute_sobol_from_tt_cores</c>.
    /// </summary>
    /// <param name="cores">
    /// TT coefficient cores in storage order. Each core has shape
    /// (r_left, n_k, r_right). cores[0] has r_left=1; cores[d-1] has r_right=1.
    /// </param>
    /// <returns>
    /// <see cref="SobolResult"/> with arrays keyed by storage-frame indices (0..d-1).
    /// The caller is responsible for translating storage-frame → user-frame if
    /// a non-identity dim-order is in effect.
    /// </returns>
    internal static SobolResult ComputeSobolFromTtCores(TensorTrainKernel.TtCore[] cores)
    {
        int d = cores.Length;
        double pi = Math.PI;

        // Per-dim Chebyshev inner-product weights:
        // w[k][0] = π,  w[k][p ≥ 1] = π/2
        var wFull = new double[d][];
        for (int k = 0; k < d; k++)
        {
            int nk = cores[k].NNodes;
            wFull[k] = new double[nk];
            for (int p = 0; p < nk; p++) wFull[k][p] = pi / 2.0;
            wFull[k][0] = pi;
        }

        // ---- constant term c_0 (all-zero multi-index)
        // v = product of cores[k][:, 0, :] chained left-to-right from [1.0]
        double[] v = new double[cores[0].RLeft];
        v[0] = 1.0;
        for (int k = 0; k < d; k++)
        {
            var A = cores[k];
            int rL = A.RLeft, rR = A.RRight;
            var vnew = new double[rR];
            for (int j = 0; j < rR; j++)
            {
                double acc = 0.0;
                for (int i = 0; i < rL; i++) acc += v[i] * A[i, 0, j];
                vnew[j] = acc;
            }
            v = vnew;
        }
        double c0 = v[0];  // scalar since rRight of last core == 1
        double constantWeightedSquared = c0 * c0 * Math.Pow(pi, d);
        double variance = ContractNonconstantWeightedSquared(cores, wFull);
        double totalWeightedSquared = constantWeightedSquared + variance;

        if (IsNumericalZeroVariance(variance, totalWeightedSquared))
        {
            return new SobolResult(new double[d], new double[d], 0.0);
        }

        // ---- Precompute left partial matrices L[0..d] and right partial matrices R[0..d]
        // L[0] = [[1]], L[k+1] = einsum("ij,ipa,jpb->ab", L[k], Aw_k, A_k)
        // R[d] = [[1]], R[k]   = einsum("ab,ipa,jpb->ij", R[k+1], Aw_k, A_k)
        var L = new double[d + 1][,];
        L[0] = new double[1, 1];
        L[0][0, 0] = 1.0;
        for (int k = 0; k < d; k++)
        {
            var A = cores[k];
            int rL = A.RLeft, nk = A.NNodes, rR = A.RRight;
            var Lnew = new double[rR, rR];
            for (int a = 0; a < rR; a++)
                for (int b = 0; b < rR; b++)
                {
                    double acc = 0.0;
                    for (int i = 0; i < rL; i++)
                        for (int j = 0; j < rL; j++)
                        {
                            double lij = L[k][i, j];
                            for (int p = 0; p < nk; p++)
                                acc += lij * A[i, p, a] * wFull[k][p] * A[j, p, b];
                        }
                    Lnew[a, b] = acc;
                }
            L[k + 1] = Lnew;
        }

        var R = new double[d + 1][,];
        R[d] = new double[1, 1];
        R[d][0, 0] = 1.0;
        for (int k = d - 1; k >= 0; k--)
        {
            var A = cores[k];
            int rL = A.RLeft, nk = A.NNodes, rR = A.RRight;
            var Rnew = new double[rL, rL];
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < rL; j++)
                {
                    double acc = 0.0;
                    for (int a = 0; a < rR; a++)
                        for (int b = 0; b < rR; b++)
                        {
                            double rab = R[k + 1][a, b];
                            for (int p = 0; p < nk; p++)
                                acc += rab * A[i, p, a] * wFull[k][p] * A[j, p, b];
                        }
                    Rnew[i, j] = acc;
                }
            R[k] = Rnew;
        }

        var firstOrder = new double[d];
        var totalOrder = new double[d];

        for (int j = 0; j < d; j++)
        {
            // ---- first-order energy[j]:
            // alpha_j >= 1 AND alpha_k = 0 for all k != j
            // left[r_j] = chain cores[k][:,0,:] for k < j (left @ core)
            var left = new double[cores[j].RLeft];
            left[0] = 1.0;
            for (int k = 0; k < j; k++)
            {
                var A = cores[k];
                int rL = A.RLeft, rR = A.RRight;
                var lnew = new double[rR];
                for (int b = 0; b < rR; b++)
                {
                    double acc = 0.0;
                    for (int i = 0; i < rL; i++) acc += left[i] * A[i, 0, b];
                    lnew[b] = acc;
                }
                left = lnew;
            }

            // right[r_{j+1}] = chain cores[k][:,0,:] for k > j (core @ right), right-to-left
            var right = new double[cores[j].RRight];
            right[0] = 1.0;
            for (int k = d - 1; k > j; k--)
            {
                var A = cores[k];
                int rL = A.RLeft, rR = A.RRight;
                var rnew = new double[rL];
                for (int i = 0; i < rL; i++)
                {
                    double acc = 0.0;
                    for (int b = 0; b < rR; b++) acc += A[i, 0, b] * right[b];
                    rnew[i] = acc;
                }
                right = rnew;
            }

            var Gj = cores[j];
            int rLj = Gj.RLeft, njNodes = Gj.NNodes, rRj = Gj.RRight;
            double sumSquared = 0.0;
            for (int m = 1; m < njNodes; m++)  // skip m=0 (constant slice)
            {
                // coef_m = left @ G_j[:, m, :] @ right  (scalar)
                double coefM = 0.0;
                for (int i = 0; i < rLj; i++)
                {
                    double lGi = 0.0;
                    for (int a = 0; a < rRj; a++) lGi += Gj[i, m, a] * right[a];
                    coefM += left[i] * lGi;
                }
                sumSquared += coefM * coefM;
            }
            // weight: (π/2) * π^(d-1)
            double weightFirst = (pi / 2.0) * Math.Pow(pi, d - 1);
            firstOrder[j] = sumSquared * weightFirst / variance;

            // ---- total-order energy[j]:
            // sum every weighted coefficient where alpha_j > 0. Computing this
            // directly avoids cancellation in total - alpha_j_zero for functions
            // with a large constant offset and tiny but valid sensitivity signal.
            int rLjj = cores[j].RLeft, rRjj = cores[j].RRight;
            double sumAlphaJPositive = 0.0;
            for (int i = 0; i < rLjj; i++)
                for (int jj = 0; jj < rLjj; jj++)
                {
                    double lij = L[j][i, jj];
                    for (int a = 0; a < rRjj; a++)
                        for (int b = 0; b < rRjj; b++)
                        {
                            double rab = R[j + 1][a, b];
                            for (int p = 1; p < cores[j].NNodes; p++)
                                sumAlphaJPositive += lij * cores[j][i, p, a] * wFull[j][p] * cores[j][jj, p, b] * rab;
                        }
                }
            totalOrder[j] = sumAlphaJPositive / variance;
        }

        return new SobolResult(firstOrder, totalOrder, variance);
    }

    /// <summary>
    /// Compute first- and total-order Sobol sensitivity indices from a Chebyshev
    /// coefficient tensor. Throws on NaN/Inf in input. Returns zero-filled
    /// <see cref="SobolResult"/> for constant functions (Variance == 0).
    /// </summary>
    internal static SobolResult ComputeSobolFromCoeffs(double[] coeffs, int[] shape)
    {
        for (int i = 0; i < coeffs.Length; i++)
            if (!double.IsFinite(coeffs[i]))
                throw new ArgumentException(
                    "coefficients contain NaN or Inf; SobolIndices() requires finite spectral coefficients");

        int nDim = shape.Length;
        var firstOrder = new double[nDim];
        var totalOrder = new double[nDim];
        double variance = 0;
        double totalWeightedSquared = 0;

        for (long flat = 0; flat < coeffs.Length; flat++)
        {
            var alpha = UnravelIndex(flat, shape);
            int nonzeroCount = 0;
            int firstNonzeroDim = -1;
            for (int d = 0; d < nDim; d++)
                if (alpha[d] > 0) { nonzeroCount++; if (firstNonzeroDim == -1) firstNonzeroDim = d; }

            double c = coeffs[flat];
            if (c == 0) continue;
            double energy = c * c * MultiIndexNormSquared(alpha);
            totalWeightedSquared += energy;
            if (nonzeroCount == 0) continue;  // skip α = 0 (mean term).
            variance += energy;
            for (int d = 0; d < nDim; d++)
                if (alpha[d] > 0) totalOrder[d] += energy;
            if (nonzeroCount == 1) firstOrder[firstNonzeroDim] += energy;
        }

        if (IsNumericalZeroVariance(variance, totalWeightedSquared))
            return new SobolResult(new double[nDim], new double[nDim], variance);
        for (int d = 0; d < nDim; d++)
        {
            firstOrder[d] /= variance;
            totalOrder[d] /= variance;
        }
        return new SobolResult(firstOrder, totalOrder, variance);
    }
}
