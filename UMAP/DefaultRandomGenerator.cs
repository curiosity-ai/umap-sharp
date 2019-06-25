using System;
using System.Runtime.CompilerServices;

namespace UMAP
{
    public sealed class DefaultRandomGenerator : IProvideRandomValues
    {
        public static DefaultRandomGenerator Instance { get; } = new DefaultRandomGenerator();
        private DefaultRandomGenerator() { }

        public bool IsThreadSafe => true;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next(int minValue, int maxValue) => ThreadSafeFastRandom.Next(minValue, maxValue);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextFloat() => ThreadSafeFastRandom.NextFloat();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void NextFloats(Span<float> buffer) => ThreadSafeFastRandom.NextFloats(buffer);
    }
}