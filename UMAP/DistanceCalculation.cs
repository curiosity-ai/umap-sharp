namespace UMAP
{
    public delegate float DistanceCalculation<RawVectorArrayDataPoint>(RawVectorArrayDataPoint x, RawVectorArrayDataPoint y) where RawVectorArrayDataPoint : IUmapDataPoint;
}