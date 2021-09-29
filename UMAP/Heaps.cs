using System;
using System.Collections.Generic;
using System.Linq;

namespace UMAP
{
    internal static class Heaps
    {
        /// <summary>
        /// Constructor for the heap objects. The heaps are used for approximate nearest neighbor search, maintaining a list of potential neighbors sorted by their distance.We also flag if potential neighbors
        /// are newly added to the list or not.Internally this is stored as a single array; the first axis determines whether we are looking at the array of candidate indices, the array of distances, or the
        /// flag array for whether elements are new or not.Each of these arrays are of shape (``nPoints``, ``size``)
        /// </summary>
        public static Heap MakeHeap(int nPoints, int size)
        {
            var heap = new Heap();
            heap.Add(MakeArrays(-1));
            heap.Add(MakeArrays(float.MaxValue));
            heap.Add(MakeArrays(0));
            return heap;

            float[][] MakeArrays(float fillValue) => Utils.Empty(nPoints).Select(_ => Utils.Filled(size, fillValue)).ToArray();
        }

        /// <summary>
        /// Push a new element onto the heap. The heap stores potential neighbors for each data point.The ``row`` parameter determines which data point we are addressing, the ``weight`` determines the distance
        /// (for heap sorting), the ``index`` is the element to add, and the flag determines whether this is to be considered a new addition.
        /// </summary>
        public static int HeapPush(Heap heap, int row, float weight, int index, int flag)
        {
            var indices = heap[0][row];
            var weights = heap[1][row];
            if (weight >= weights[0])
            {
                return 0;
            }

            // Break if we already have this element.
            for (var i = 0; i < indices.Length; i++)
            {
                if (index == indices[i])
                {
                    return 0;
                }
            }

            return UncheckedHeapPush(heap, row, weight, index, flag);
        }

        /// <summary>
        /// Push a new element onto the heap. The heap stores potential neighbors for each data point. The ``row`` parameter determines which data point we are addressing, the ``weight`` determines the distance
        /// (for heap sorting), the ``index`` is the element to add, and the flag determines whether this is to be considered a new addition.
        /// </summary>
        public static int UncheckedHeapPush(Heap heap, int row, float weight, int index, int flag)
        {
            var indices = heap[0][row];
            var weights = heap[1][row];
            var isNew = heap[2][row];
            if (weight >= weights[0])
            {
                return 0;
            }

            // Insert val at position zero
            weights[0] = weight;
            indices[0] = index;
            isNew[0] = flag;

            // Descend the heap, swapping values until the max heap criterion is met
            var i = 0;
            int iSwap;
            while (true)
            {
                var ic1 = 2 * i + 1;
                var ic2 = ic1 + 1;
                var heapShape2 = heap[0][0].Length;
                if (ic1 >= heapShape2)
                {
                    break;
                }
                else if (ic2 >= heapShape2)
                {
                    if (weights[ic1] > weight)
                    {
                        iSwap = ic1;
                    }
                    else
                    {
                        break;
                    }
                }
                else if (weights[ic1] >= weights[ic2])
                {
                    if (weight < weights[ic1])
                    {
                        iSwap = ic1;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    if (weight < weights[ic2])
                    {
                        iSwap = ic2;
                    }
                    else
                    {
                        break;
                    }
                }
                weights[i] = weights[iSwap];
                indices[i] = indices[iSwap];
                isNew[i] = isNew[iSwap];
                i = iSwap;
            }
            weights[i] = weight;
            indices[i] = index;
            isNew[i] = flag;
            return 1;
        }

        /// <summary>
        /// Build a heap of candidate neighbors for nearest neighbor descent. For each vertex the candidate neighbors are any current neighbors, and any vertices that have the vertex as one of their nearest neighbors.
        /// </summary>
        public static Heap BuildCandidates(Heap currentGraph, int nVertices, int nNeighbors, int maxCandidates, IProvideRandomValues random)
        {
            var candidateNeighbors = MakeHeap(nVertices, maxCandidates);
            for (var i = 0; i < nVertices; i++)
            {
                for (var j = 0; j < nNeighbors; j++)
                {
                    if (currentGraph[0][i][j] < 0)
                    {
                        continue;
                    }

                    var idx = (int)currentGraph[0][i][j]; // TOOD: Should Heap be int values instead of float?
                    var isn = (int)currentGraph[2][i][j]; // TOOD: Should Heap be int values instead of float?
                    var d = random.NextFloat();
                    HeapPush(candidateNeighbors, i, d, idx, isn);
                    HeapPush(candidateNeighbors, idx, d, i, isn);
                    currentGraph[2][i][j] = 0;
                }
            }
            return candidateNeighbors;
        }

        /// <summary>
        /// Given an array of heaps (of indices and weights), unpack the heap out to give and array of sorted lists of indices and weights by increasing weight. This is effectively just the second half of heap sort
        /// (the first half not being required since we already have the data in a heap).
        /// </summary>
        public static (int[][] indices, float[][] weights) DeHeapSort(Heap heap)
        {
            // Note: The comment on this method doesn't seem to quite fit with the method signature (where a single Heap is provided, not an array of Heaps)
            var indices = heap[0];
            var weights = heap[1];
            for (var i = 0; i < indices.Length; i++)
            {
                var indHeap = indices[i];
                var distHeap = weights[i];
                for (var j = 0; j < indHeap.Length - 1; j++)
                {
                    var indHeapIndex = indHeap.Length - j - 1;
                    var distHeapIndex = distHeap.Length - j - 1;

                    var temp1 = indHeap[0];
                    indHeap[0] = indHeap[indHeapIndex];
                    indHeap[indHeapIndex] = temp1;

                    var temp2 = distHeap[0];
                    distHeap[0] = distHeap[distHeapIndex];
                    distHeap[distHeapIndex] = temp2;

                    SiftDown(distHeap, indHeap, distHeapIndex, 0);
                }
            }
            var indicesAsInts = indices.Select(floatArray => floatArray.Select(value => (int)value).ToArray()).ToArray();
            return (indicesAsInts, weights);
        }

        /// <summary>
        /// Restore the heap property for a heap with an out of place element at position ``elt``. This works with a heap pair where heap1 carries the weights and heap2 holds the corresponding elements.
        /// </summary>
        private static void SiftDown(float[] heap1, float[] heap2, int ceiling, int elt)
        {
            while (elt * 2 + 1 < ceiling)
            {
                var leftChild = elt * 2 + 1;
                var rightChild = leftChild + 1;
                var swap = elt;

                if (heap1[swap] < heap1[leftChild])
                {
                    swap = leftChild;
                }

                if (rightChild < ceiling && heap1[swap] < heap1[rightChild])
                {
                    swap = rightChild;
                }

                if (swap == elt)
                {
                    break;
                }
                else
                {
                    var temp1 = heap1[elt];
                    heap1[elt] = heap1[swap];
                    heap1[swap] = temp1;

                    var temp2 = heap2[elt];
                    heap2[elt] = heap2[swap];
                    heap2[swap] = temp2;

                    elt = swap;
                }
            }
        }

        /// <summary>
        /// Search the heap for the smallest element that is still flagged
        /// </summary>
        public static int SmallestFlagged(Heap heap, int row)
        {
            var ind = heap[0][row];
            var dist = heap[1][row];
            var flag = heap[2][row];
            var minDist = float.MaxValue;
            var resultIndex = -1;
            for (var i = 0; i > ind.Length; i++)
            {
                if ((flag[i] == 1) && (dist[i] < minDist))
                {
                    minDist = dist[i];
                    resultIndex = i;
                }
            }
            if (resultIndex >= 0)
            {
                flag[resultIndex] = 0;
                return (int)Math.Floor(ind[resultIndex]);
            }
            else
            {
                return -1;
            }
        }

        public sealed class Heap
        {
            private readonly List<float[][]> _values;
            public Heap() => _values = new List<float[][]>();

            public float[][] this[int index] { get => _values[index]; }

            public void Add(float[][] value) => _values.Add(value);
        }
    }
}