using ChebyshevSharp.Internal;

namespace ChebyshevSharp.Tests;

public class TensorShapeTests
{
    [Fact]
    public void CheckedProduct_Returns_Product_For_Valid_Shape()
    {
        long product = TensorShape.CheckedProduct(new[] { 3, 5, 7 }, "grid");

        Assert.Equal(105L, product);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-2)]
    public void CheckedProduct_Rejects_Non_Positive_Dimensions(int value)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            TensorShape.CheckedProduct(new[] { 3, value, 7 }, "grid"));

        Assert.Contains("grid", ex.Message);
        Assert.Contains(value.ToString(), ex.Message);
    }

    [Fact]
    public void CheckedProduct_Throws_When_Product_Exceeds_Long_MaxValue()
    {
        var ex = Assert.Throws<OverflowException>(() =>
            TensorShape.CheckedProduct(
                new[] { int.MaxValue, int.MaxValue, int.MaxValue },
                "full grid"));

        Assert.Contains("full grid", ex.Message);
        Assert.Contains(int.MaxValue.ToString(), ex.Message);
    }

    [Fact]
    public void TryCheckedProduct_Returns_True_When_Product_Fits()
    {
        bool fits = TensorShape.TryCheckedProduct(new[] { 3, 5, 7 }, out long product);

        Assert.True(fits);
        Assert.Equal(105L, product);
    }

    [Fact]
    public void TryCheckedProduct_Returns_False_When_Product_Exceeds_Long_MaxValue()
    {
        bool fits = TensorShape.TryCheckedProduct(
            new[] { int.MaxValue, int.MaxValue, int.MaxValue },
            out long product);

        Assert.False(fits);
        Assert.Equal(long.MaxValue, product);
    }

    [Fact]
    public void ProductAtMost_Returns_Product_When_At_Or_Below_Cap()
    {
        long product = TensorShape.ProductAtMost(new[] { 4, 5, 6 }, 120);

        Assert.Equal(120L, product);
    }

    [Fact]
    public void ProductAtMost_Returns_Cap_Plus_One_When_Product_Would_Exceed_Cap()
    {
        long product = TensorShape.ProductAtMost(new[] { 4, 5, 7 }, 120);

        Assert.Equal(121L, product);
    }

    [Fact]
    public void CheckedByteSize_Throws_When_Element_Size_Product_Overflows()
    {
        long elements = long.MaxValue / 8 + 1;

        var ex = Assert.Throws<OverflowException>(() =>
            TensorShape.CheckedByteSize(elements, 8, "ToDense"));

        Assert.Contains("ToDense", ex.Message);
    }

    [Fact]
    public void RequireArrayLength_Returns_Int_Length_When_Fit_For_Array()
    {
        int length = TensorShape.RequireArrayLength(1024, "GetEvaluationPoints");

        Assert.Equal(1024, length);
    }

    [Fact]
    public void RequireArrayLength_Throws_When_Length_Exceeds_Int_MaxValue()
    {
        var ex = Assert.Throws<OverflowException>(() =>
            TensorShape.RequireArrayLength((long)int.MaxValue + 1, "GetEvaluationPoints"));

        Assert.Contains("GetEvaluationPoints", ex.Message);
        Assert.Contains("int.MaxValue", ex.Message);
    }
}
