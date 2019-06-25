using System;
using System.Linq;
using Xunit;
using static UMAP.UnitTests.UnitTestData;

namespace UMAP.UnitTests
{
    public static class UmapTests
    {
        [Fact]
        public static void StepMethod2D()
        {
            var umap = new Umap(random: new DeterministicRandomGenerator(42));
            var nEpochs = umap.InitializeFit(TestData);
            for (var i = 0; i < nEpochs; i++)
                umap.Step();
            var embedding = umap.GetEmbedding();
            Assert.Equal(500, nEpochs);
            AssertNestedFloatArraysEquivalent(TestResults2D, embedding);
        }

        [Fact]
        public static void StepMethod3D()
        {
            var umap = new Umap(random: new DeterministicRandomGenerator(42), dimensions: 3);
            var nEpochs = umap.InitializeFit(TestData);
            for (var i = 0; i < nEpochs; i++)
                umap.Step();
            var embedding = umap.GetEmbedding();
            Assert.Equal(500, nEpochs);
            AssertNestedFloatArraysEquivalent(TestResults3D, embedding);
        }

        [Fact]
        public static void FindsNearestNeighbors()
        {
            var nNeighbors = 10;
            var umap = new Umap(random: new DeterministicRandomGenerator(42), numberOfNeighbors: nNeighbors);
            var (knnIndices, knnDistances) = umap.NearestNeighbors(TestData);

            Assert.Equal(knnDistances.Length, TestData.Length);
            Assert.Equal(knnIndices.Length, TestData.Length);

            Assert.Equal(knnDistances[0].Length, nNeighbors);
            Assert.Equal(knnIndices[0].Length, nNeighbors);
        }

        [Fact]
        public static void FindsABParamsUsingLevenbergMarquardtForDefaultSettings()
        {
            const float expectedA = 1.5769434603113077f;
            const float expectedB = 0.8950608779109733f;

            var (a, b) = Umap.FindABParams(1, 0.1f);
            Assert.True(AreCloseEnough(a, expectedA));
            Assert.True(AreCloseEnough(b, expectedB));

            bool AreCloseEnough(float x, float y) => Math.Abs(x - y) < 0.01;
        }

        private static void AssertNestedFloatArraysEquivalent(float[][] expected, float[][] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            foreach (var (expectedRow, actualRow) in expected.Zip(actual, (expectedRow, actualRow) => (expectedRow, actualRow)))
            {
                Assert.Equal(expectedRow.Length, actualRow.Length);
                foreach (var (expectedValue, actualValue) in expectedRow.Zip(actualRow, (expectedValue, actualValue) => (expectedValue, actualValue)))
                    Assert.True(Math.Abs(expectedValue - actualValue) < 1e-5);
            }
        }
    }
}