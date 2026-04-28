using Xunit;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Serializes all Console.SetOut-capturing tests into a single xUnit collection
/// so they do not race on Console.Out when the test runner parallelizes classes.
/// </summary>
[CollectionDefinition("ConsoleCapture")]
public class ConsoleCaptureCollection : ICollectionFixture<object>
{
}
