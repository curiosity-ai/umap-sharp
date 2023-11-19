using System;
using System.Collections.Generic;
using System.Text;

namespace UMAP
{
    public class RawVectorArrayDataPoint : IUmapDataPoint
    {
        /// <summary>
        /// Creates new instance of <see cref="RawVectorArrayDataPoint"/>.
        /// </summary>
        /// <param name="data"></param>
        public RawVectorArrayDataPoint(float[] data)
        {
            Data = data;
        }

        public float[] Data { get; private set; }

        /// <summary>
        /// Define an implicit conversion operator from <see cref="float[]"/>.
        /// </summary>
        public static implicit operator RawVectorArrayDataPoint(float[] data) => new RawVectorArrayDataPoint(data);

        /// <summary>
        /// Implicit conversation back to <see cref="float[]"/>.
        /// </summary>
        public static implicit operator float[](RawVectorArrayDataPoint x) => x.Data;
    }
}
