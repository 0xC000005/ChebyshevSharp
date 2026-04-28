namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train algebra (addition, scalar mul, negation,
/// TT-SVD rounding, inner product). Operates on flat <see cref="TensorTrainKernel.TtCore"/>
/// arrays. Members are added incrementally across Phase 2 Tasks 4 (InnerProduct),
/// 9 (scalar algebra), and 10 (binary algebra + rounding).
/// </summary>
internal static class TensorTrainAlgebra
{
}
