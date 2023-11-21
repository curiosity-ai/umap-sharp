namespace UMAP
{
    public delegate float DistanceCalculation<T>(T x, T y) where T : IUmapDataPoint;
}