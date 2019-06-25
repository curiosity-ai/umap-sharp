namespace UMAP.UnitTests
{
    // See https://github.com/zeh/prando/blob/master/src/Prando.ts
    public sealed class Prando
    {
        private readonly int _seed;
        private int _value;
        public Prando(int seed)
        {
            _seed = GetSafeSeed(seed);
            _value = _seed;
        }

        /// <summary>
        /// Generates a pseudo-random number between a lower (inclusive) and a higher (exclusive) bounds
        /// </summary>
        public float NextFloat(float min = 0, float pseudoMax = 1)
        {
            Recalculate();
            return Map(_value, int.MinValue, int.MaxValue, min, pseudoMax);
        }

        /// <summary>
        /// Returns a random integer that is within a specified range.
        /// </summary>
        /// <param name="minValue">The inclusive lower bound of the random number returned.</param>
        /// <param name="maxValue">The exclusive upper bound of the random number returned. maxValue must be greater than or equal to minValue.</param>
        /// <returns>A 32-bit signed integer greater than or equal to minValue and less than maxValue; that is, the range of return values includes minValue but not maxValue. If minValue
        //  equals maxValue, minValue is returned.</returns>
        public int Next(int minValue, int maxValue)
        {
            Recalculate();
            return (int)Map(_value, int.MinValue, int.MaxValue, minValue, maxValue);
        }

        private void Recalculate()
        {
            _value = XorShift(_value);
        }

        private static float Map(int val, int minFrom, int maxFrom, float minTo, float maxTo)
        {
            var availableRange = (float)maxFrom - minFrom; // Perform the calculation as float because it will overflow if it's done in Int32 space
            var distanceOfValueIntoRange = (float)val - minFrom;
            return (distanceOfValueIntoRange / availableRange) * (maxTo - minTo) + minTo;
        }

        private static int XorShift(int value)
        {
            // Xorshift*32
            // Based on George Marsaglia's work: http://www.jstatsoft.org/v08/i14/paper
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            return value;
        }

        private static int GetSafeSeed(int seed) => (seed == 0) ? 1 : seed;
   }
}