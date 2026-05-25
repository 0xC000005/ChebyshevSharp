using System.Diagnostics;
using ChebyshevSharp;

namespace CallableBondSurrogate;

public sealed class CallableAnchoredHdmrSurrogate
{
    private readonly double[] _anchor;
    private readonly double _anchorValue;
    private readonly OneDimensionalComponent[] _oneDimensionalComponents;
    private readonly PairComponent[] _pairComponents;

    private CallableAnchoredHdmrSurrogate(
        double[] anchor,
        double anchorValue,
        OneDimensionalComponent[] oneDimensionalComponents,
        PairComponent[] pairComponents,
        int buildEvaluations,
        double buildSeconds)
    {
        _anchor = anchor;
        _anchorValue = anchorValue;
        _oneDimensionalComponents = oneDimensionalComponents;
        _pairComponents = pairComponents;
        BuildEvaluations = buildEvaluations;
        BuildSeconds = buildSeconds;
    }

    public int BuildEvaluations { get; }

    public double BuildSeconds { get; }

    public int OneDimensionalComponentCount => _oneDimensionalComponents.Length;

    public int PairComponentCount => _pairComponents.Length;

    public static CallableAnchoredHdmrSurrogate Build(
        CallableBondFullDimensionalWrapper wrapper,
        int oneDimensionalNodes = 9,
        int pairNodes = 5)
    {
        ArgumentNullException.ThrowIfNull(wrapper);
        if (oneDimensionalNodes < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(oneDimensionalNodes), "At least three nodes are required.");
        }

        if (pairNodes < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(pairNodes), "At least three nodes are required.");
        }

        double[] anchor = wrapper.CreateBasePoint();
        double anchorValue = wrapper.Price(anchor);
        double[][] domain = CallableRiskAcceptance.BuildPublicDomain();
        var oneDimensionalComponents = new OneDimensionalComponent[CallableBondFullDimensionalWrapper.DimensionCount];
        var pairComponents = new List<PairComponent>();
        int buildEvaluations = 0;

        Stopwatch sw = Stopwatch.StartNew();
        for (int dimension = 0; dimension < oneDimensionalComponents.Length; dimension++)
        {
            int capturedDimension = dimension;
            var approximation = new ChebyshevApproximation(
                (point, _) => wrapper.Price(WithCoordinate(anchor, capturedDimension, point[0])),
                numDimensions: 1,
                domain: [domain[capturedDimension]],
                nNodes: [oneDimensionalNodes]);
            approximation.Build(verbose: false);
            buildEvaluations += approximation.NEvaluations;
            oneDimensionalComponents[dimension] = new OneDimensionalComponent(capturedDimension, approximation);
        }

        foreach ((int First, int Second) pair in BuildPairs())
        {
            int first = pair.First;
            int second = pair.Second;
            var approximation = new ChebyshevApproximation(
                (point, _) => wrapper.Price(WithCoordinates(anchor, first, point[0], second, point[1])),
                numDimensions: 2,
                domain: [domain[first], domain[second]],
                nNodes: [pairNodes, pairNodes]);
            approximation.Build(verbose: false);
            buildEvaluations += approximation.NEvaluations;
            pairComponents.Add(new PairComponent(first, second, approximation));
        }

        sw.Stop();
        return new CallableAnchoredHdmrSurrogate(
            (double[])anchor.Clone(),
            anchorValue,
            oneDimensionalComponents,
            pairComponents.ToArray(),
            buildEvaluations,
            sw.Elapsed.TotalSeconds);
    }

    public double Eval(double[] point)
    {
        if (point.Length != CallableBondFullDimensionalWrapper.DimensionCount)
        {
            throw new ArgumentException(
                $"Expected {CallableBondFullDimensionalWrapper.DimensionCount} coordinates.",
                nameof(point));
        }

        var oneDimensionalRaw = new double[_oneDimensionalComponents.Length];
        double value = _anchorValue;
        foreach (OneDimensionalComponent component in _oneDimensionalComponents)
        {
            double raw = component.Eval(point[component.Dimension]);
            oneDimensionalRaw[component.Dimension] = raw;
            value += raw - _anchorValue;
        }

        foreach (PairComponent component in _pairComponents)
        {
            double raw = component.Eval(point[component.FirstDimension], point[component.SecondDimension]);
            value += raw
                - oneDimensionalRaw[component.FirstDimension]
                - oneDimensionalRaw[component.SecondDimension]
                + _anchorValue;
        }

        return value;
    }

    private static IReadOnlyList<(int First, int Second)> BuildPairs()
    {
        var pairs = new HashSet<(int First, int Second)>();
        for (int i = 0; i < CallableBondFullDimensionalWrapper.CurveBumpCount; i++)
        {
            AddPair(pairs, i, CallableRiskAcceptance.CouponDimension);
            AddPair(pairs, i, CallableRiskAcceptance.CallPriceDimension);
            AddPair(pairs, i, CallableRiskAcceptance.SigmaDimension);
            if (i + 1 < CallableBondFullDimensionalWrapper.CurveBumpCount)
            {
                AddPair(pairs, i, i + 1);
            }
        }

        AddPair(pairs, CallableRiskAcceptance.CallPriceDimension, CallableRiskAcceptance.SigmaDimension);
        AddPair(pairs, CallableRiskAcceptance.CouponDimension, CallableRiskAcceptance.SigmaDimension);
        AddPair(pairs, CallableRiskAcceptance.MaturityDimension, CallableRiskAcceptance.SigmaDimension);
        AddPair(pairs, CallableRiskAcceptance.FirstCallDimension, CallableRiskAcceptance.SigmaDimension);
        AddPair(pairs, CallableRiskAcceptance.MaturityDimension, CallableRiskAcceptance.FirstCallDimension);
        AddPair(pairs, CallableRiskAcceptance.MaturityDimension, CallableRiskAcceptance.CallPriceDimension);
        AddPair(pairs, CallableRiskAcceptance.FirstCallDimension, CallableRiskAcceptance.CallPriceDimension);

        return pairs
            .OrderBy(pair => pair.First)
            .ThenBy(pair => pair.Second)
            .ToArray();
    }

    private static void AddPair(HashSet<(int First, int Second)> pairs, int first, int second)
    {
        if (first == second)
        {
            return;
        }

        if (first > second)
        {
            (first, second) = (second, first);
        }

        pairs.Add((first, second));
    }

    private static double[] WithCoordinate(double[] anchor, int dimension, double value)
    {
        double[] point = (double[])anchor.Clone();
        point[dimension] = value;
        return point;
    }

    private static double[] WithCoordinates(
        double[] anchor,
        int firstDimension,
        double firstValue,
        int secondDimension,
        double secondValue)
    {
        double[] point = (double[])anchor.Clone();
        point[firstDimension] = firstValue;
        point[secondDimension] = secondValue;
        return point;
    }

    private sealed record OneDimensionalComponent(
        int Dimension,
        ChebyshevApproximation Approximation)
    {
        public double Eval(double value)
            => Approximation.VectorizedEval([value], [0]);
    }

    private sealed record PairComponent(
        int FirstDimension,
        int SecondDimension,
        ChebyshevApproximation Approximation)
    {
        public double Eval(double firstValue, double secondValue)
            => Approximation.VectorizedEval([firstValue, secondValue], [0, 0]);
    }
}
