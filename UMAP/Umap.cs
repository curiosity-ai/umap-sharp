using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using static UMAP.NNDescent;
using static UMAP.Tree;

namespace UMAP
{
    public sealed class Umap
    {
        private const float SMOOTH_K_TOLERANCE = 1e-5f;
        private const float MIN_K_DIST_SCALE = 1e-3f;

        private readonly float _learningRate = 1f;
        private readonly float _localConnectivity = 1f;
        private readonly float _minDist = 0.1f;
        private readonly int _negativeSampleRate = 5;
        private readonly float _repulsionStrength = 1;
        private readonly float _setOpMixRatio = 1;
        private readonly float _spread = 1;

        private readonly DistanceCalculation _distanceFn;
        private readonly IProvideRandomValues _random;
        private readonly int _nNeighbors;
        private readonly int? _customNumberOfEpochs;
        private readonly ProgressReporter _progressReporter;

        // KNN state (can be precomputed and supplied via initializeFit)
        private int[][] _knnIndices = null;
        private float[][] _knnDistances = null;

        // Internal graph connectivity representation
        private SparseMatrix _graph = null;
        private float[][] _x = null;
        private bool _isInitialized = false;
        private FlatTree[] _rpForest = new FlatTree[0];

        // Projected embedding
        private float[] _embedding;
        private readonly OptimizationState _optimizationState;

        /// <summary>
        /// The progress will be a value from 0 to 1 that indicates approximately how much of the processing has been completed
        /// </summary>
        public delegate void ProgressReporter(float progress);

        public Umap(
            DistanceCalculation distance = null,
            IProvideRandomValues random = null,
            int dimensions = 2,
            int numberOfNeighbors = 15,
            int? customNumberOfEpochs = null,
            ProgressReporter progressReporter = null)
        {
            if ((customNumberOfEpochs != null) && (customNumberOfEpochs <= 0))
            {
                throw new ArgumentOutOfRangeException(nameof(customNumberOfEpochs), "if non-null then must be a positive value");
            }

            _distanceFn = distance ?? DistanceFunctions.Cosine;
            _random = random ?? DefaultRandomGenerator.Instance;
            _nNeighbors = numberOfNeighbors;
            _optimizationState = new OptimizationState { Dim = dimensions };
            _customNumberOfEpochs = customNumberOfEpochs;
            _progressReporter = progressReporter;
        }

        /// <summary>
        /// Initializes fit by computing KNN and a fuzzy simplicial set, as well as initializing the projected embeddings. Sets the optimization state ahead of optimization steps.
        /// Returns the number of epochs to be used for the SGD optimization.
        /// </summary>
        public int InitializeFit(float[][] x)
        {
            // We don't need to reinitialize if we've already initialized for this data
            if ((_x == x) && _isInitialized)
            {
                return GetNEpochs();
            }

            // For large quantities of data (which is where the progress estimating is more useful), InitializeFit takes at least 80% of the total time (the calls to Step are
            // completed much more quickly AND they naturally lend themselves to granular progress updates; one per loop compared to the recommended number of epochs)
            ProgressReporter initializeFitProgressReporter = (_progressReporter is null) ? (progress => { }) : ScaleProgressReporter(_progressReporter, 0, 0.8f);

            _x = x;
            if ((_knnIndices is null) && (_knnDistances is null))
            {
                // This part of the process very roughly accounts for 1/3 of the work
                (_knnIndices, _knnDistances) = NearestNeighbors(x, ScaleProgressReporter(initializeFitProgressReporter, 0, 0.3f));
            }

            // This part of the process very roughly accounts for 2/3 of the work (the reamining work is in the Step calls)
            _graph = FuzzySimplicialSet(x, _nNeighbors, _setOpMixRatio, ScaleProgressReporter(initializeFitProgressReporter, 0.3f, 1));

            var (head, tail, epochsPerSample) = InitializeSimplicialSetEmbedding();

            // Set the optimization routine state
            _optimizationState.Head = head;
            _optimizationState.Tail = tail;
            _optimizationState.EpochsPerSample = epochsPerSample;

            // Now, initialize the optimization steps
            InitializeOptimization();
            PrepareForOptimizationLoop();
            _isInitialized = true;

            return GetNEpochs();
        }

        public float[][] GetEmbedding()
        {
            var final = new float[_optimizationState.NVertices][];
            Span<float> span = _embedding.AsSpan();
            for (int i = 0; i < _optimizationState.NVertices; i++)
            {
                final[i] = span.Slice(i * _optimizationState.Dim, _optimizationState.Dim).ToArray();
            }
            return final;
        }

        /// <summary>
        /// Gets the number of epochs for optimizing the projection - NOTE: This heuristic differs from the python version
        /// </summary>
        private int GetNEpochs()
        {
            if (_customNumberOfEpochs != null)
            {
                return _customNumberOfEpochs.Value;
            }

            var length = _graph.Dims.rows;
            if (length <= 2500)
            {
                return 500;
            }
            else if (length <= 5000)
            {
                return 400;
            }
            else if (length <= 7500)
            {
                return 300;
            }
            else
            {
                return 200;
            }
        }

        /// <summary>
        /// Compute the ``nNeighbors`` nearest points for each data point in ``X`` - this may be exact, but more likely is approximated via nearest neighbor descent.
        /// </summary>
        internal (int[][] knnIndices, float[][] knnDistances) NearestNeighbors(float[][] x, ProgressReporter progressReporter)
        {
            var metricNNDescent = MakeNNDescent(_distanceFn, _random);
            progressReporter(0.05f);
            var nTrees = 5 + Round(Math.Sqrt(x.Length) / 20);
            var nIters = Math.Max(5, (int)Math.Floor(Math.Round(Math.Log(x.Length, 2))));
            progressReporter(0.1f);
            var leafSize = Math.Max(10, _nNeighbors);
            var forestProgressReporter = ScaleProgressReporter(progressReporter, 0.1f, 0.4f);
            _rpForest = Enumerable.Range(0, nTrees)
                .Select(i =>
                {
                    forestProgressReporter((float)i / nTrees);
                    return FlattenTree(MakeTree(x, leafSize, i, _random), leafSize);
                })
                .ToArray();
            var leafArray = MakeLeafArray(_rpForest);
            progressReporter(0.45f);
            var nnDescendProgressReporter = ScaleProgressReporter(progressReporter, 0.5f, 1);
            return metricNNDescent(x, leafArray, _nNeighbors, nIters, startingIteration: (i, max) => nnDescendProgressReporter((float)i / max));

            // Handle python3 rounding down from 0.5 discrpancy
            int Round(double n) => (n == 0.5) ? 0 : (int)Math.Floor(Math.Round(n));
        }

        /// <summary>
        /// Given a set of data X, a neighborhood size, and a measure of distance compute the fuzzy simplicial set(here represented as a fuzzy graph in the form of a sparse matrix) associated
        /// to the data. This is done by locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local fuzzy
        /// simplicial sets into a global one via a fuzzy union.
        /// </summary>
        private SparseMatrix FuzzySimplicialSet(float[][] x, int nNeighbors, float setOpMixRatio, ProgressReporter progressReporter)
        {
            var knnIndices = _knnIndices ?? new int[0][];
            var knnDistances = _knnDistances ?? new float[0][];
            progressReporter(0.1f);
            var (sigmas, rhos) = SmoothKNNDistance(knnDistances, nNeighbors, _localConnectivity);
            progressReporter(0.2f);
            var (rows, cols, vals) = ComputeMembershipStrengths(knnIndices, knnDistances, sigmas, rhos);
            progressReporter(0.3f);
            var sparseMatrix = new SparseMatrix(rows, cols, vals, (x.Length, x.Length));
            var transpose = sparseMatrix.Transpose();
            var prodMatrix = sparseMatrix.PairwiseMultiply(transpose);
            progressReporter(0.4f);
            var a = sparseMatrix.Add(transpose).Subtract(prodMatrix);
            progressReporter(0.5f);
            var b = a.MultiplyScalar(setOpMixRatio);
            progressReporter(0.6f);
            var c = prodMatrix.MultiplyScalar(1 - setOpMixRatio);
            progressReporter(0.7f);
            var result = b.Add(c);
            progressReporter(0.8f);
            return result;
        }

        private static (float[] sigmas, float[] rhos) SmoothKNNDistance(float[][] distances, int k, float localConnectivity = 1, int nIter = 64, float bandwidth = 1)
        {
            var target = Math.Log(k, 2) * bandwidth; // TODO: Use Math.Log2 (when update framework to a version that supports it) or consider a pre-computed table
            var rho = new float[distances.Length];
            var result = new float[distances.Length];
            for (var i = 0; i < distances.Length; i++)
            {
                var lo = 0f;
                var hi = float.MaxValue;
                var mid = 1f;

                // TODO[umap-js]: This is very inefficient, but will do for now. FIXME
                var ithDistances = distances[i];
                var nonZeroDists = ithDistances.Where(d => d > 0).ToArray();
                if (nonZeroDists.Length >= localConnectivity)
                {
                    var index = (int)Math.Floor(localConnectivity);
                    var interpolation = localConnectivity - index;
                    if (index > 0)
                    {
                        rho[i] = nonZeroDists[index - 1];
                        if (interpolation > SMOOTH_K_TOLERANCE)
                        {
                            rho[i] += interpolation * (nonZeroDists[index] - nonZeroDists[index - 1]);
                        }
                    }
                    else
                    {
                        rho[i] = interpolation * nonZeroDists[0];
                    }
                }
                else if (nonZeroDists.Length > 0)
                {
                    rho[i] = Utils.Max(nonZeroDists);
                }

                for (var n = 0; n < nIter; n++)
                {
                    var psum = 0.0;
                    for (var j = 1; j < distances[i].Length; j++)
                    {
                        var d = distances[i][j] - rho[i];
                        if (d > 0)
                        {
                            psum += Math.Exp(-(d / mid));
                        }
                        else
                        {
                            psum += 1.0;
                        }
                    }
                    if (Math.Abs(psum - target) < SMOOTH_K_TOLERANCE)
                    {
                        break;
                    }

                    if (psum > target)
                    {
                        hi = mid;
                        mid = (lo + hi) / 2;
                    }
                    else
                    {
                        lo = mid;
                        if (hi == float.MaxValue)
                        {
                            mid *= 2;
                        }
                        else
                        {
                            mid = (lo + hi) / 2;
                        }
                    }
                }

                result[i] = mid;

                // TODO[umap-js]: This is very inefficient, but will do for now. FIXME
                if (rho[i] > 0)
                {
                    var meanIthDistances = Utils.Mean(ithDistances);
                    if (result[i] < MIN_K_DIST_SCALE * meanIthDistances)
                    {
                        result[i] = MIN_K_DIST_SCALE * meanIthDistances;
                    }
                }
                else
                {
                    var meanDistances = Utils.Mean(distances.Select(Utils.Mean).ToArray());
                    if (result[i] < MIN_K_DIST_SCALE * meanDistances)
                    {
                        result[i] = MIN_K_DIST_SCALE * meanDistances;
                    }
                }
            }
            return (result, rho);
        }

        private static (int[] rows, int[] cols, float[] vals) ComputeMembershipStrengths(int[][] knnIndices, float[][] knnDistances, float[] sigmas, float[] rhos)
        {
            var nSamples = knnIndices.Length;
            var nNeighbors = knnIndices[0].Length;

            var rows = new int[nSamples * nNeighbors];
            var cols = new int[nSamples * nNeighbors];
            var vals = new float[nSamples * nNeighbors];
            for (var i = 0; i < nSamples; i++)
            {
                for (var j = 0; j < nNeighbors; j++)
                {
                    if (knnIndices[i][j] == -1)
                    {
                        continue; // We didn't get the full knn for i
                    }

                    float val;
                    if (knnIndices[i][j] == i)
                    {
                        val = 0;
                    }
                    else if (knnDistances[i][j] - rhos[i] <= 0.0)
                    {
                        val = 1;
                    }
                    else
                    {
                        val = (float)Math.Exp(-((knnDistances[i][j] - rhos[i]) / sigmas[i]));
                    }

                    rows[i * nNeighbors + j] = i;
                    cols[i * nNeighbors + j] = knnIndices[i][j];
                    vals[i * nNeighbors + j] = val;
                }
            }
            return (rows, cols, vals);
        }

        /// <summary>
        /// Initialize a fuzzy simplicial set embedding, using a specified initialisation method and then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and low
        /// dimensional fuzzy simplicial sets.
        /// </summary>
        private (int[] head, int[] tail, float[] epochsPerSample) InitializeSimplicialSetEmbedding()
        {
            var nEpochs = GetNEpochs();
            var graphMax = 0f;
            foreach (var value in _graph.GetValues())
            {
                if (graphMax < value)
                {
                    graphMax = value;
                }
            }

            var graph = _graph.Map(value => (value < graphMax / nEpochs) ? 0 : value);

            // We're not computing the spectral initialization in this implementation until we determine a better eigenvalue/eigenvector computation approach

            _embedding = new float[graph.Dims.rows * _optimizationState.Dim];
            SIMDint.Uniform(ref _embedding, 10, _random);

            // Get graph data in ordered way...
            var weights = new List<float>();
            var head = new List<int>();
            var tail = new List<int>();
            foreach (var (row, col, value) in graph.GetAll())
            {
                if (value != 0)
                {
                    weights.Add(value);
                    tail.Add(row);
                    head.Add(col);
                }
            }
            ShuffleTogether(head, tail, weights);
            return (head.ToArray(), tail.ToArray(), MakeEpochsPerSample(weights.ToArray(), nEpochs));
        }

        private void ShuffleTogether<T, T2, T3>(List<T> list, List<T2> other, List<T3> weights)
        {
            int n = list.Count;
            if (other.Count != n) { throw new Exception(); }
            while (n > 1)
            {
                n--;
                int k = _random.Next(0, n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;

                T2 otherValue = other[k];
                other[k] = other[n];
                other[n] = otherValue;

                T3 weightsValue = weights[k];
                weights[k] = weights[n];
                weights[n] = weightsValue;
            }
        }

        private static float[] MakeEpochsPerSample(float[] weights, int nEpochs)
        {
            var result = Utils.Filled(weights.Length, -1);
            var max = Utils.Max(weights);
            foreach (var (n, i) in weights.Select((w, i) => ((w / max) * nEpochs, i)))
            {
                if (n > 0)
                {
                    result[i] = nEpochs / n;
                }
            }
            return result;
        }

        private void InitializeOptimization()
        {
            // Initialized in initializeSimplicialSetEmbedding()
            var head = _optimizationState.Head;
            var tail = _optimizationState.Tail;
            var epochsPerSample = _optimizationState.EpochsPerSample;

            var nEpochs = GetNEpochs();
            var nVertices = _graph.Dims.cols;

            var (a, b) = FindABParams(_spread, _minDist);

            _optimizationState.Head = head;
            _optimizationState.Tail = tail;
            _optimizationState.EpochsPerSample = epochsPerSample;
            _optimizationState.A = a;
            _optimizationState.B = b;
            _optimizationState.NEpochs = nEpochs;
            _optimizationState.NVertices = nVertices;
        }

        internal static (float a, float b) FindABParams(float spread, float minDist)
        {
            // 2019-06-21 DWR: If we need to support other spread, minDist values then we might be able to use the LM implementation in Accord.NET but I'll hard code values that relate to the default configuration for now
            if ((spread != 1) || (minDist != 0.1f))
            {
                throw new ArgumentException($"Currently, the {nameof(FindABParams)} method only supports spread, minDist values of 1, 0.1 (the Levenberg-Marquardt algorithm is required to process other values");
            }

            return (1.5694704762346365f, 0.8941996053733949f);
        }

        private void PrepareForOptimizationLoop()
        {
            // Hyperparameters
            var repulsionStrength = _repulsionStrength;
            var learningRate = _learningRate;
            var negativeSampleRate = _negativeSampleRate;

            var epochsPerSample = _optimizationState.EpochsPerSample;

            var dim = _optimizationState.Dim;

            var epochsPerNegativeSample = epochsPerSample.Select(e => e / negativeSampleRate).ToArray();
            var epochOfNextNegativeSample = epochsPerNegativeSample.ToArray();
            var epochOfNextSample = epochsPerSample.ToArray();

            _optimizationState.EpochOfNextSample = epochOfNextSample;
            _optimizationState.EpochOfNextNegativeSample = epochOfNextNegativeSample;
            _optimizationState.EpochsPerNegativeSample = epochsPerNegativeSample;

            _optimizationState.MoveOther = true;
            _optimizationState.InitialAlpha = learningRate;
            _optimizationState.Alpha = learningRate;
            _optimizationState.Gamma = repulsionStrength;
            _optimizationState.Dim = dim;
        }

        /// <summary>
        /// Manually step through the optimization process one epoch at a time
        /// </summary>
        public int Step()
        {
            var currentEpoch = _optimizationState.CurrentEpoch;
            var numberOfEpochsToComplete = GetNEpochs();
            if (currentEpoch < numberOfEpochsToComplete)
            {
                OptimizeLayoutStep(currentEpoch);
                if (_progressReporter is object)
                {
                    // InitializeFit roughly approximately takes 80% of the processing time for large quantities of data, leaving 20% for the Step iterations - the progress reporter
                    // calls made here are based on the assumption that Step will be called the recommended number of times (the number-of-epochs value returned from InitializeFit)
                    ScaleProgressReporter(_progressReporter, 0.8f, 1)((float)currentEpoch / numberOfEpochsToComplete);
                }
            }
            return _optimizationState.CurrentEpoch;
        }

        /// <summary>
        /// Improve an embedding using stochastic gradient descent to minimize the fuzzy set cross entropy between the 1-skeletons of the high dimensional and low dimensional fuzzy simplicial sets.
        /// In practice this is done by sampling edges based on their membership strength(with the (1-p) terms coming from negative sampling similar to word2vec).
        /// </summary>
        private void OptimizeLayoutStep(int n)
        {
            if (_random.IsThreadSafe)
            {
                Parallel.For(0, _optimizationState.EpochsPerSample.Length, Iterate);
            }
            else
            {
                for (var i = 0; i < _optimizationState.EpochsPerSample.Length; i++)
                {
                    Iterate(i);
                }
            }

            _optimizationState.Alpha = _optimizationState.InitialAlpha * (1f - ((float)n / _optimizationState.NEpochs));
            _optimizationState.CurrentEpoch += 1;

            void Iterate(int i)
            {
                if (_optimizationState.EpochOfNextSample[i] >= n)
                {
                    return;
                }

                Span<float> embeddingSpan = _embedding.AsSpan();

                int j = _optimizationState.Head[i];
                int k = _optimizationState.Tail[i];

                var current = embeddingSpan.Slice(j * _optimizationState.Dim, _optimizationState.Dim);
                var other = embeddingSpan.Slice(k * _optimizationState.Dim, _optimizationState.Dim);

                var distSquared = RDist(current, other);
                var gradCoeff = 0f;

                if (distSquared > 0)
                {
                    gradCoeff = -2 * _optimizationState.A * _optimizationState.B * (float)Math.Pow(distSquared, _optimizationState.B - 1);
                    gradCoeff /= _optimizationState.A * (float)Math.Pow(distSquared, _optimizationState.B) + 1;
                }

                const float clipValue = 4f;
                for (var d = 0; d < _optimizationState.Dim; d++)
                {
                    var gradD = Clip(gradCoeff * (current[d] - other[d]), clipValue);
                    current[d] += gradD * _optimizationState.Alpha;
                    if (_optimizationState.MoveOther)
                    {
                        other[d] += -gradD * _optimizationState.Alpha;
                    }
                }

                _optimizationState.EpochOfNextSample[i] += _optimizationState.EpochsPerSample[i];

                var nNegSamples = (int)Math.Floor((double)(n - _optimizationState.EpochOfNextNegativeSample[i]) / _optimizationState.EpochsPerNegativeSample[i]);

                for (var p = 0; p < nNegSamples; p++)
                {
                    k = _random.Next(0, _optimizationState.NVertices);
                    other = embeddingSpan.Slice(k * _optimizationState.Dim, _optimizationState.Dim);
                    distSquared = RDist(current, other);
                    gradCoeff = 0f;
                    if (distSquared > 0)
                    {
                        gradCoeff = 2 * _optimizationState.Gamma * _optimizationState.B;
                        gradCoeff *= _optimizationState.GetDistanceFactor(distSquared); //Preparation for future work for interpolating the table before optimizing
                    }
                    else if (j == k)
                    {
                        continue;
                    }

                    for (var d = 0; d < _optimizationState.Dim; d++)
                    {
                        var gradD = 4f;
                        if (gradCoeff > 0)
                        {
                            gradD = Clip(gradCoeff * (current[d] - other[d]), clipValue);
                        }

                        current[d] += gradD * _optimizationState.Alpha;
                    }
                }

                _optimizationState.EpochOfNextNegativeSample[i] += nNegSamples * _optimizationState.EpochsPerNegativeSample[i];
            }
        }

        /// <summary>
        /// Reduced Euclidean distance
        /// </summary>
        private static float RDist(Span<float> x, Span<float> y)
        {
            //return Mosaik.Core.SIMD.Euclidean(ref x, ref y);
            var distSquared = 0f;
            for (var i = 0; i < x.Length; i++)
            {
                var d = x[i] - y[i];
                distSquared += d * d;
            }
            return distSquared;
        }

        /// <summary>
        /// Standard clamping of a value into a fixed range
        /// </summary>
        private static float Clip(float x, float clipValue)
        {
            if (x > clipValue)
            {
                return clipValue;
            }
            else if (x < -clipValue)
            {
                return -clipValue;
            }
            else
            {
                return x;
            }
        }

        private static ProgressReporter ScaleProgressReporter(ProgressReporter progressReporter, float start, float end)
        {
            var range = end - start;
            return progress => progressReporter((range * progress) + start);
        }

        public static class DistanceFunctions
        {
            public static float Cosine(float[] lhs, float[] rhs)
            {
                return 1 - (SIMD.DotProduct(ref lhs, ref rhs) / (SIMD.Magnitude(ref lhs) * SIMD.Magnitude(ref rhs)));
            }

            public static float CosineForNormalizedVectors(float[] lhs, float[] rhs)
            {
                return 1 - SIMD.DotProduct(ref lhs, ref rhs);
            }

            public static float Euclidean(float[] lhs, float[] rhs)
            {
                return (float)Math.Sqrt(SIMD.Euclidean(ref lhs, ref rhs)); // TODO: Replace with netcore3 MathF class when the framework is available
            }
        }

        private sealed class OptimizationState
        {
            public int CurrentEpoch                 =  0;
            public int[] Head                       =  new int[0];
            public int[] Tail                       =  new int[0];
            public float[] EpochsPerSample          =  new float[0];
            public float[] EpochOfNextSample        =  new float[0];
            public float[] EpochOfNextNegativeSample=  new float[0];
            public float[] EpochsPerNegativeSample  =  new float[0];
            public bool MoveOther                   =  true;
            public float InitialAlpha               =  1;
            public float Alpha                      =  1;
            public float Gamma                      =  1;
            public float A                          =  1.5769434603113077f;
            public float B                          =  0.8950608779109733f;
            public int Dim                          =  2;
            public int NEpochs                      =  500;
            public int NVertices                    =  0;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public float GetDistanceFactor(float distSquared) => 1f / ((0.001f + distSquared) * (float)(A * Math.Pow(distSquared, B) + 1));
        }
    }
}
