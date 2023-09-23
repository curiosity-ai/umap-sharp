namespace UMAP
{
    public delegate float DistanceCalculation<T>(IUmapDistanceParameter<T>[] x, IUmapDistanceParameter<T>[] y);
}