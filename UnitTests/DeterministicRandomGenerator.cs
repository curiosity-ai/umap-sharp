using System;

namespace UMAP.UnitTests
{
    public sealed class DeterministicRandomGenerator : IProvideRandomValues
    {
        private readonly Prando _rnd;
        public DeterministicRandomGenerator(int seed) => _rnd = new Prando(seed);

        public bool IsThreadSafe => false;

        public int Next(int minValue, int maxValue) => _rnd.Next(minValue, maxValue);

        public float NextFloat() => _rnd.NextFloat();

        public void NextFloats(Span<float> buffer)
        {
            for (var i = 0; i < buffer.Length; i++)
                buffer[i] = _rnd.NextFloat();
        }
    }
}