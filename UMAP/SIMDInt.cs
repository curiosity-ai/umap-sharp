using System.Numerics;
using System.Runtime.CompilerServices;

namespace UMAP
{
    internal static class SIMDint
    {
        private static readonly int _vs1 = Vector<int>.Count;
        private static readonly int _vs2 = 2 * Vector<int>.Count;
        private static readonly int _vs3 = 3 * Vector<int>.Count;
        private static readonly int _vs4 = 4 * Vector<int>.Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Zero(ref int[] lhs)
        {
            var count = lhs.Length;
            var offset = 0;

            while (count >= _vs4)
            {
                Vector<int>.Zero.CopyTo(lhs, offset);
                Vector<int>.Zero.CopyTo(lhs, offset + _vs1);
                Vector<int>.Zero.CopyTo(lhs, offset + _vs2);
                Vector<int>.Zero.CopyTo(lhs, offset + _vs3);
                if (count == _vs4)
                {
                    return;
                }

                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                Vector<int>.Zero.CopyTo(lhs, offset);
                Vector<int>.Zero.CopyTo(lhs, offset + _vs1);
                if (count == _vs2)
                {
                    return;
                }

                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                Vector<int>.Zero.CopyTo(lhs, offset);
                if (count == _vs1)
                {
                    return;
                }

                count -= _vs1;
                offset += _vs1;
            }
            if (count > 0)
            {
                while (count > 0)
                {
                    lhs[offset] = 0;
                    offset++; count--;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Uniform(ref float[] data, float a, IProvideRandomValues random)
        {
            float a2 = 2 * a;
            float an = -a;
            random.NextFloats(data);
            SIMD.Multiply(ref data, a2);
            SIMD.Add(ref data, an);
        }
    }
}