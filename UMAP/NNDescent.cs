using System;
using System.Collections.Generic;
using static UMAP.Heaps;

namespace UMAP
{
    internal static class NNDescent<T>
    {
        public delegate (int[][] indices, float[][] weights) NNDescentFn(
            IUmapDistanceParameter<T>[][] data,
            int[][] leafArray,
            int nNeighbors,
            int nIters = 10,
            int maxCandidates = 50,
            float delta = 0.001f,
            float rho = 0.5f,
            bool rpTreeInit = true,
            Action<int, int> startingIteration = null
        );

        /// <summary>
        /// Create a version of nearest neighbor descent.
        /// </summary>
        public static NNDescentFn MakeNNDescent(DistanceCalculation<T> distanceFn, IProvideRandomValues random)
        {
            return (data, leafArray, nNeighbors, nIters, maxCandidates, delta, rho, rpTreeInit, startingIteration) =>
            {
                var nVertices = data.Length;
                var currentGraph = MakeHeap(data.Length, nNeighbors);
                for (var i = 0; i < data.Length; i++)
                {
                    var indices = Utils.RejectionSample(nNeighbors, data.Length, random);
                    for (var j = 0; j < indices.Length; j++)
                    {
                        var d = distanceFn(data[i], data[indices[j]]);
                        HeapPush(currentGraph, i, d, indices[j], 1);
                        HeapPush(currentGraph, indices[j], d, i, 1);
                    }
                }
                if (rpTreeInit)
                {
                    for (var n = 0; n < leafArray.Length; n++)
                    {
                        for (var i = 0; i < leafArray[n].Length; i++)
                        {
                            if (leafArray[n][i] < 0)
                            {
                                break;
                            }

                            for (var j = i + 1; j < leafArray[n].Length; j++)
                            {
                                if (leafArray[n][j] < 0)
                                {
                                    break;
                                }

                                var d = distanceFn(data[leafArray[n][i]], data[leafArray[n][j]]);
                                HeapPush(currentGraph, leafArray[n][i], d, leafArray[n][j], 1);
                                HeapPush(currentGraph, leafArray[n][j], d, leafArray[n][i], 1);
                            }
                        }
                    }
                }
                for (var n = 0; n < nIters; n++)
                {
                    startingIteration?.Invoke(n, nIters);
                    var candidateNeighbors = BuildCandidates(currentGraph, nVertices, nNeighbors, maxCandidates, random);
                    var c = 0;
                    for (var i = 0; i < nVertices; i++)
                    {
                        for (var j = 0; j < maxCandidates; j++)
                        {
                            var p = (int)Math.Floor(candidateNeighbors[0][i][j]);
                            if ((p < 0) || (random.NextFloat() < rho))
                            {
                                continue;
                            }

                            for (var k = 0; k < maxCandidates; k++)
                            {
                                var q = (int)Math.Floor(candidateNeighbors[0][i][k]);
                                var cj = candidateNeighbors[2][i][j];
                                var ck = candidateNeighbors[2][i][k];
                                if (q < 0 || ((cj == 0) && (ck == 0)))
                                {
                                    continue;
                                }

                                var d = distanceFn(data[p], data[q]);
                                c += HeapPush(currentGraph, p, d, q, 1);
                                c += HeapPush(currentGraph, q, d, p, 1);
                            }
                        }
                    }
                    if (c <= delta * nNeighbors * data.Length)
                    {
                        break;
                    }
                }
                return DeHeapSort(currentGraph);
            };
        }
    }
}