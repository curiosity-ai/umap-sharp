namespace UMAP
{
    public delegate float DistanceCalculation<T>(IUmapDistance<T>[] x, IUmapDistance<T>[] y);
}