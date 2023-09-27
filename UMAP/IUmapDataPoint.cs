namespace UMAP
{
    /// <summary>
    /// Represents a single data point to be processed by <see cref="Umap{T}"/>.
    /// </summary>
    public interface IUmapDataPoint
    {
        /// <summary>
        /// The data being operated on.
        /// </summary>
        float[] Data { get; }
    }
}