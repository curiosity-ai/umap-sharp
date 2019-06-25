using System.Collections.Generic;
using Xunit;

namespace UMAP.UnitTests
{
    public static class SparseMatrixTests
    {
        [Fact]
        public static void ConstructsSpareMatrixFromRowsColsVals()
        {
            var rows = new[] { 0, 0, 1, 1 };
            var cols = new[] { 0, 1, 0, 1 };
            var vals = new[] { 1f, 2f, 3f, 4f };
            var dims = (2, 2);
            var matrix = new SparseMatrix(rows, cols, vals, dims);
            Assert.Equal(rows, matrix.GetRows());
            Assert.Equal(cols, matrix.GetCols());
            Assert.Equal(vals, matrix.GetValues());
            Assert.Equal(2, matrix.Dims.rows);
            Assert.Equal(2, matrix.Dims.cols);
        }

        [Fact]
        public static void HasGetSetMethods()
        {
            var matrix = GetTestMatrix();
            Assert.Equal(2, matrix.Get(0, 1));
            matrix.Set(0, 1, 9);
            Assert.Equal(9, matrix.Get(0, 1));
        }

        [Fact]
        public static void HasMapMethod()
        {
            var matrix = GetTestMatrix();
            var newMatrix = matrix.Map(value => value + 1);
            Assert.Equal(new[] { new[] { 2f, 3f }, new[] { 4f, 5f } }, newMatrix.ToArray());
        }

        [Fact]
        public static void HasForEachMethod()
        {
            var rows = new[] { 0, 1 };
            var cols = new[] { 0, 0 };
            var vals = new[] { 1f, 3f };
            var dims = (2, 2);
            var matrix = new SparseMatrix(rows, cols, vals, dims);
            var entries = new List<float[]>();
            matrix.ForEach((value, row, col) => entries.Add(new float[] { value, row, col }));
            Assert.Equal(new[] { new[] { 1f, 0f, 0f }, new[] { 3f, 1f, 0f } }, entries.ToArray());
        }

        [Fact]
        public static void TransposeMethod() => Assert.Equal(new[] { new[] { 1f, 3f }, new[] { 2f, 4f } }, GetTestMatrix().Transpose().ToArray());

        [Fact]
        public static void PairwiseMultiplyMethod() => Assert.Equal(new[] { new[] { 1f, 4f }, new[] { 9f, 16f } }, GetTestMatrix().PairwiseMultiply(GetTestMatrix()).ToArray());

        [Fact]
        public static void AddMethod() => Assert.Equal(new[] { new[] { 2f, 4f }, new[] { 6f, 8f } }, GetTestMatrix().Add(GetTestMatrix()).ToArray());

        [Fact]
        public static void SubtractMethod() => Assert.Equal(new[] { new[] { 0f, 0f }, new[] { 0f, 0f } }, GetTestMatrix().Subtract(GetTestMatrix()).ToArray());

        [Fact]
        public static void ScalarMultiplyMethod() => Assert.Equal(new[] { new[] { 3f, 6f }, new[] { 9f, 12f } }, GetTestMatrix().MultiplyScalar(3).ToArray());

        [Fact]
        public static void GetCSRMethod()
        {
            var (indices, values, indptr) = GetNormalizationTestMatrix().GetCSR();
            Assert.Equal(new[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 }, indices);
            Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }, values);
            Assert.Equal(new[] { 0, 3, 6 }, indptr);
        }

        private static SparseMatrix GetNormalizationTestMatrix()
        {
            var rows = new[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
            var cols = new[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
            var vals = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
            var dims = (3, 3);
            return new SparseMatrix(rows, cols, vals, dims);
        }

        private static SparseMatrix GetTestMatrix()
        {
            var rows = new[] { 0, 0, 1, 1 };
            var cols = new[] { 0, 1, 0, 1 };
            var vals = new[] { 1f, 2f, 3f, 4f };
            var dims = (2, 2);
            return new SparseMatrix(rows, cols, vals, dims);
        }
    }
}