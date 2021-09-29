using System;
using System.Collections.Generic;
using System.Linq;

namespace UMAP
{
    internal static class Tree
    {
        /// <summary>
        /// Construct a random projection tree based on ``data`` with leaves of size at most ``leafSize``
        /// </summary>
        public static RandomProjectionTreeNode MakeTree(float[][] data, int leafSize, int n, IProvideRandomValues random)
        {
            var indices = Enumerable.Range(0, data.Length).ToArray();
            return MakeEuclideanTree(data, indices, leafSize, n, random);
        }

        private static RandomProjectionTreeNode MakeEuclideanTree(float[][] data, int[] indices, int leafSize, int q, IProvideRandomValues random)
        {
            if (indices.Length > leafSize)
            {
                var (indicesLeft, indicesRight, hyperplaneVector, hyperplaneOffset) = EuclideanRandomProjectionSplit(data, indices, random);
                var leftChild = MakeEuclideanTree(data, indicesLeft, leafSize, q + 1, random);
                var rightChild = MakeEuclideanTree(data, indicesRight, leafSize, q + 1, random);
                return new RandomProjectionTreeNode { Indices = indices, LeftChild = leftChild, RightChild = rightChild, IsLeaf = false, Hyperplane = hyperplaneVector, Offset = hyperplaneOffset };
            }
            else
            {
                return new RandomProjectionTreeNode { Indices = indices, LeftChild = null, RightChild = null, IsLeaf = true, Hyperplane = Array.Empty<float>(), Offset = 0 };
            }
        }

        public static FlatTree FlattenTree(RandomProjectionTreeNode tree, int leafSize)
        {
            var nNodes = NumNodes(tree);
            var nLeaves = NumLeaves(tree);

            // TODO[umap-js]: Verify that sparse code is not relevant...
            var hyperplanes = Utils.Range(nNodes).Select(_ => new float[tree.Hyperplane.Length]).ToArray();

            var offsets = new float[nNodes];
            var children = Utils.Range(nNodes).Select(_ => new[] { -1, -1 }).ToArray();
            var indices = Utils.Range(nLeaves).Select(_ => Utils.Range(leafSize).Select(___ => -1).ToArray()).ToArray();
            RecursiveFlatten(tree, hyperplanes, offsets, children, indices, 0, 0);
            return new FlatTree { Hyperplanes = hyperplanes, Offsets = offsets, Children = children, Indices = indices };
        }

        /// <summary>
        /// Given a set of ``indices`` for data points from ``data``, create a random hyperplane to split the data, returning two arrays indices that fall on either side of the hyperplane. This is
        /// the basis for a random projection tree, which simply uses this splitting recursively. This particular split uses euclidean distance to determine the hyperplane and which side each data
        /// sample falls on.
        /// </summary>
        private static (int[] indicesLeft, int[] indicesRight, float[] hyperplaneVector, float hyperplaneOffset) EuclideanRandomProjectionSplit(float[][] data, int[] indices, IProvideRandomValues random)
        {
            var dim = data[0].Length;

            // Select two random points, set the hyperplane between them
            var leftIndex = random.Next(0, indices.Length);
            var rightIndex = random.Next(0, indices.Length);
            rightIndex += (leftIndex == rightIndex) ? 1 : 0;
            rightIndex %= indices.Length;
            var left = indices[leftIndex];
            var right = indices[rightIndex];

            // Compute the normal vector to the hyperplane (the vector between the two points) and the offset from the origin
            var hyperplaneOffset = 0f;
            var hyperplaneVector = new float[dim];
            for (var i = 0; i < hyperplaneVector.Length; i++)
            {
                hyperplaneVector[i] = data[left][i] - data[right][i];
                hyperplaneOffset -= (hyperplaneVector[i] * (data[left][i] + data[right][i])) / 2;
            }

            // For each point compute the margin (project into normal vector)
            // If we are on lower side of the hyperplane put in one pile, otherwise put it in the other pile (if we hit hyperplane on the nose, flip a coin)
            var nLeft = 0;
            var nRight = 0;
            var side = new int[indices.Length];
            for (var i = 0; i < indices.Length; i++)
            {
                var margin = hyperplaneOffset;
                for (var d = 0; d < dim; d++)
                {
                    margin += hyperplaneVector[d] * data[indices[i]][d];
                }

                if (margin == 0)
                {
                    side[i] = random.Next(0, 2);
                    if (side[i] == 0)
                    {
                        nLeft += 1;
                    }
                    else
                    {
                        nRight += 1;
                    }
                }
                else if (margin > 0)
                {
                    side[i] = 0;
                    nLeft += 1;
                }
                else
                {
                    side[i] = 1;
                    nRight += 1;
                }
            }

            // Now that we have the counts, allocate arrays
            var indicesLeft = new int[nLeft];
            var indicesRight = new int[nRight];

            // Populate the arrays with indices according to which side they fell on
            nLeft = 0;
            nRight = 0;
            for (var i = 0; i < side.Length; i++)
            {
                if (side[i] == 0)
                {
                    indicesLeft[nLeft] = indices[i];
                    nLeft += 1;
                }
                else
                {
                    indicesRight[nRight] = indices[i];
                    nRight += 1;
                }
            }

            return (indicesLeft, indicesRight, hyperplaneVector, hyperplaneOffset);
        }

        private static (int nodeNum, int leafNum) RecursiveFlatten(RandomProjectionTreeNode tree, float[][] hyperplanes, float[] offsets, int[][] children, int[][] indices, int nodeNum, int leafNum)
        {
            if (tree.IsLeaf)
            {
                children[nodeNum][0] = -leafNum;

                // TODO[umap-js]: Triple check this operation corresponds to
                // indices[leafNum : tree.indices.shape[0]] = tree.indices
                tree.Indices.CopyTo(indices[leafNum], 0);
                leafNum += 1;
                return (nodeNum, leafNum);
            }
            else
            {
                hyperplanes[nodeNum] = tree.Hyperplane;
                offsets[nodeNum] = tree.Offset;
                children[nodeNum][0] = nodeNum + 1;
                var oldNodeNum = nodeNum;

                var res = RecursiveFlatten(
                    tree.LeftChild,
                    hyperplanes,
                    offsets,
                    children,
                    indices,
                    nodeNum + 1,
                    leafNum
                );
                nodeNum = res.nodeNum;
                leafNum = res.leafNum;

                children[oldNodeNum][1] = nodeNum + 1;

                res = RecursiveFlatten(
                    tree.RightChild,
                    hyperplanes,
                    offsets,
                    children,
                    indices,
                    nodeNum + 1,
                    leafNum
                );
                return (res.nodeNum, res.leafNum);
            }
        }

        private static int NumNodes(RandomProjectionTreeNode tree) => tree.IsLeaf ? 1 : (1 + NumNodes(tree.LeftChild) + NumNodes(tree.RightChild));

        private static int NumLeaves(RandomProjectionTreeNode tree) => tree.IsLeaf ? 1 : (1 + NumLeaves(tree.LeftChild) + NumLeaves(tree.RightChild));

        /// <summary>
        /// Generate an array of sets of candidate nearest neighbors by constructing a random projection forest and taking the leaves of all the trees. Any given tree has leaves that are
        /// a set of potential nearest neighbors.Given enough trees the set of all such leaves gives a good likelihood of getting a good set of nearest neighbors in composite. Since such
        /// a random projection forest is inexpensive to compute, this can be a useful means of seeding other nearest neighbor algorithms.
        /// </summary>
        public static int[][] MakeLeafArray(FlatTree[] forest)
        {
            if (forest.Length > 0)
            {
                var output = new List<int[]>();
                foreach (var tree in forest)
                {
                    foreach (var entry in tree.Indices)
                    {
                        output.Add(entry);
                    }
                }
                return output.ToArray();
            }
            else
            {
                return new[] { new[] { -1 } };
            }
        }

        /// <summary>
        /// Searches a flattened rp-tree for a point
        /// </summary>
        public static int[] SearchFlatTree(float[] point, FlatTree tree, IProvideRandomValues random)
        {
            var node = 0;
            while (tree.Children[node][0] > 0)
            {
                var side = SelectSide(tree.Hyperplanes[node], tree.Offsets[node], point, random);
                if (side == 0)
                {
                    node = tree.Children[node][0];
                }
                else
                {
                    node = tree.Children[node][1];
                }
            }
            var index = -1 * tree.Children[node][0];
            return tree.Indices[index];
        }

        /// <summary>
        /// Select the side of the tree to search during flat tree search
        /// </summary>
        private static int SelectSide(float[] hyperplane, float offset, float[] point, IProvideRandomValues random)
        {
            var margin = offset;
            for (var d = 0; d < point.Length; d++)
            {
                margin += hyperplane[d] * point[d];
            }

            if (margin == 0)
            {
                return random.Next(0, 2);
            }
            else if (margin > 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        public sealed class FlatTree
        {
            public float[][] Hyperplanes { get; set; }
            public float[] Offsets { get; set; }
            public int[][] Children { get; set; }
            public int[][] Indices { get; set; }
        }

        public sealed class RandomProjectionTreeNode
        {
            public bool IsLeaf { get; set; }
            public int[] Indices { get; set; }
            public RandomProjectionTreeNode LeftChild { get; set; }
            public RandomProjectionTreeNode RightChild { get; set; }
            public float[] Hyperplane { get; set; }
            public float Offset { get; set; }
        }
    }
}
