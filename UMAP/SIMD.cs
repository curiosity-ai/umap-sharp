using System;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace UMAP
{
    internal static class SIMD<T>
    {
        private static readonly int _vs1 = Vector<float>.Count;
        private static readonly int _vs2 = 2 * Vector<float>.Count;
        private static readonly int _vs3 = 3 * Vector<float>.Count;
        private static readonly int _vs4 = 4 * Vector<float>.Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Magnitude(ref IUmapDistance<T>[] vec) => (float)Math.Sqrt(DotProduct(ref vec, ref vec));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Euclidean(ref float[] lhs, ref float[] rhs)
        {
            float result = 0f;

            var count = lhs.Length;
            var offset = 0;
            Vector<float> diff;
            while (count >= _vs4)
            {
                diff = new Vector<float>(lhs, offset) - new Vector<float>(rhs, offset); result += Vector.Dot(diff, diff);
                diff = new Vector<float>(lhs, offset + _vs1) - new Vector<float>(rhs, offset + _vs1); result += Vector.Dot(diff, diff);
                diff = new Vector<float>(lhs, offset + _vs2) - new Vector<float>(rhs, offset + _vs2); result += Vector.Dot(diff, diff);
                diff = new Vector<float>(lhs, offset + _vs3) - new Vector<float>(rhs, offset + _vs3); result += Vector.Dot(diff, diff);
                if (count == _vs4)
                {
                    return result;
                }

                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                diff = new Vector<float>(lhs, offset) - new Vector<float>(rhs, offset); result += Vector.Dot(diff, diff);
                diff = new Vector<float>(lhs, offset + _vs1) - new Vector<float>(rhs, offset + _vs1); result += Vector.Dot(diff, diff);
                if (count == _vs2)
                {
                    return result;
                }

                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                diff = new Vector<float>(lhs, offset) - new Vector<float>(rhs, offset); result += Vector.Dot(diff, diff);
                if (count == _vs1)
                {
                    return result;
                }

                count -= _vs1;
                offset += _vs1;
            }
            if (count > 0)
            {
                while (count > 0)
                {
                    var d = (lhs[offset] - rhs[offset]);
                    result += d * d;
                    offset++; count--;
                }
            }
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ref float[] lhs, float f)
        {
            var count = lhs.Length;
            var offset = 0;
            var v = new Vector<float>(f);
            while (count >= _vs4)
            {
                (new Vector<float>(lhs, offset) + v).CopyTo(lhs, offset);
                (new Vector<float>(lhs, offset + _vs1) + v).CopyTo(lhs, offset + _vs1);
                (new Vector<float>(lhs, offset + _vs2) + v).CopyTo(lhs, offset + _vs2);
                (new Vector<float>(lhs, offset + _vs3) + v).CopyTo(lhs, offset + _vs3);
                if (count == _vs4)
                {
                    return;
                }

                count -= _vs4;
                offset += _vs4;
            }
            if (count >= _vs2)
            {
                (new Vector<float>(lhs, offset) + v).CopyTo(lhs, offset);
                (new Vector<float>(lhs, offset + _vs1) + v).CopyTo(lhs, offset + _vs1);
                if (count == _vs2)
                {
                    return;
                }

                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                (new Vector<float>(lhs, offset) + v).CopyTo(lhs, offset);
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
                    lhs[offset] += f;
                    offset++; count--;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(ref float[] lhs, float f)
        {
            var count = lhs.Length;
            var offset = 0;
            while (count >= _vs4)
            {
                (new Vector<float>(lhs, offset) * f).CopyTo(lhs, offset);
                (new Vector<float>(lhs, offset + _vs1) * f).CopyTo(lhs, offset + _vs1);
                (new Vector<float>(lhs, offset + _vs2) * f).CopyTo(lhs, offset + _vs2);
                (new Vector<float>(lhs, offset + _vs3) * f).CopyTo(lhs, offset + _vs3);
                if (count == _vs4)
                {
                    return;
                }

                count -= _vs4;
                offset += _vs4;
            }
            if (count >= _vs2)
            {
                (new Vector<float>(lhs, offset) * f).CopyTo(lhs, offset);
                (new Vector<float>(lhs, offset + _vs1) * f).CopyTo(lhs, offset + _vs1);
                if (count == _vs2)
                {
                    return;
                }

                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                (new Vector<float>(lhs, offset) * f).CopyTo(lhs, offset);
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
                    lhs[offset] *= f;
                    offset++; count--;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct(ref IUmapDistance<T>[] lhs, ref IUmapDistance<T>[] rhs)
        {
            var result = 0f;
            var count = lhs.Length;
            var offset = 0;
            while (count >= _vs4)
            {
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset));
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset + _vs1), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset + _vs1));
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset + _vs2), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset + _vs2));
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset + _vs3), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset + _vs3));
                if (count == _vs4)
                {
                    return result;
                }

                count -= _vs4;
                offset += _vs4;
            }
            if (count >= _vs2)
            {
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset));
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset + _vs1), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset + _vs1));
                if (count == _vs2)
                {
                    return result;
                }

                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                result += Vector.Dot(new Vector<float>(lhs.Select(x => x.Data).ToArray(), offset), new Vector<float>(rhs.Select(x => x.Data).ToArray(), offset));
                if (count == _vs1)
                {
                    return result;
                }

                count -= _vs1;
                offset += _vs1;
            }
            if (count > 0)
            {
                while (count > 0)
                {
                    result += lhs.Select(x => x.Data).ToArray()[offset] * rhs.Select(x => x.Data).ToArray()[offset];
                    offset++; count--;
                }
            }
            return result;
        }
    }
}