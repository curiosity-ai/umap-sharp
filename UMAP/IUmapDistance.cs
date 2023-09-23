using System;
using System.Collections.Generic;
using System.Text;

namespace UMAP
{
    public interface IUmapDistanceParameter<T>
    {
        float EmbeddingVectorValue { get; set; }
    }
}
