using AmericanOptionDynamicChebyshev;

namespace ChebyshevSharp.Tests.Finance;

public sealed class AmericanOptionExampleProgramTests
{
    [Fact]
    public void Program_main_prints_reproducible_case_study_summary()
    {
        TextWriter originalOut = Console.Out;
        using var writer = new StringWriter();

        try
        {
            Console.SetOut(writer);

            Program.Main();
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        string output = writer.ToString();
        Assert.Contains("American option Dynamic Chebyshev case study", output);
        Assert.Contains("QLNet FD reference:", output);
        Assert.Contains("LSM regression:", output);
        Assert.Contains("Stanford-style LSPI:", output);
        Assert.Contains("Dynamic Chebyshev:", output);
        Assert.Contains("Online speedup:", output);
        Assert.Contains("Grid max abs err:", output);
    }
}
