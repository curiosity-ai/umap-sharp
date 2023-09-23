using System;
using System.Collections.Generic;
using System.Text;

namespace UMAP
{
    public interface IUmapDistance<T>
    {
        float Data { get; set; }
        T RelatedComplexModel { get; set; }
    }
}
