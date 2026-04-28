using System.IO;
using System.Reflection;

namespace ChebyshevSharp.Tests.Helpers;

internal static class PcbFixtures
{
    /// <summary>
    /// Returns the absolute path to a fixture file, resolved relative to the
    /// test assembly's output directory.
    /// </summary>
    public static string Path(string name)
    {
        string baseDir = System.IO.Path.GetDirectoryName(
            Assembly.GetExecutingAssembly().Location)!;
        return System.IO.Path.Combine(baseDir, "fixtures", name);
    }
}
