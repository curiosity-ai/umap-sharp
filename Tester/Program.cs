using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.IO;
using System.Linq;
using MessagePack;
using UMAP;

namespace Tester
{
    class Program
    {
        static void Main()
        {
            // Note: The MNIST data here consist of normalized vectors (so the CosineForNormalizedVectors distance function can be safely used)
            var data = MessagePackSerializer.Deserialize<LabelledVector[]>(File.ReadAllBytes("MNIST-LabelledVectorArray-60000x100.msgpack"));
            data = data.Take(10_000).ToArray();

            var timer = Stopwatch.StartNew();
            var umap = new Umap(distance: Umap.DistanceFunctions.CosineForNormalizedVectors);

            Console.WriteLine("Initialize fit..");
            var nEpochs = umap.InitializeFit(data.Select(entry => entry.Vector).ToArray());
            Console.WriteLine("- Done");
            Console.WriteLine();
            Console.WriteLine("Calculating..");
            for (var i = 0; i < nEpochs; i++)
            {
                umap.Step();
                if ((i % 10) == 0)
                    Console.WriteLine($"- Completed {i + 1} of {nEpochs}");
            }
            Console.WriteLine("- Done");
            var embeddings = umap.GetEmbedding()
                .Select(vector => new { X = vector[0], Y = vector[1] })
                .ToArray();
            timer.Stop();
            Console.WriteLine("Time taken: " + timer.Elapsed);

            // Fit the vectors to a 0-1 range (this isn't necessary if feeding these values down from a server to a browser to draw with Plotly because ronend because Plotly scales the axes to the data)
            var minX = embeddings.Min(vector => vector.X);
            var rangeX = embeddings.Max(vector => vector.X) - minX;
            var minY = embeddings.Min(vector => vector.Y);
            var rangeY = embeddings.Max(vector => vector.Y) - minY;
            var scaledEmbeddings = embeddings
                .Select(vector => new { X = (vector.X - minX) / rangeX, Y = (vector.Y - minY) / rangeY })
                .ToArray();

            const int width = 1600;
            const int height = 1200;
            using (var bitmap = new Bitmap(width, height))
            {
                using (var g = Graphics.FromImage(bitmap))
                {
                    g.FillRectangle(Brushes.DarkBlue, 0, 0, width, height);
                    g.SmoothingMode = SmoothingMode.HighQuality;
                    g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    using (var font = new Font("Tahoma", 6))
                    {
                        foreach (var (vector, uid) in scaledEmbeddings.Zip(data, (vector, entry) => (vector, entry.UID)))
                            g.DrawString(uid, font, Brushes.White, vector.X * width, vector.Y * height);
                    }
                }
                bitmap.Save("Output-Label.png");
            }

            var colors = "#006400,#00008b,#b03060,#ff4500,#ffd700,#7fff00,#00ffff,#ff00ff,#6495ed,#ffdab9"
                .Split(',')
                .Select(c => ColorTranslator.FromHtml(c))
                .Select(c => new SolidBrush(c))
                .ToArray();
            using (var bitmap = new Bitmap(width, height))
            {
                using (var g = Graphics.FromImage(bitmap))
                {
                    g.FillRectangle(Brushes.White, 0, 0, width, height);
                    g.SmoothingMode = SmoothingMode.HighQuality;
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    foreach (var (vector, uid) in scaledEmbeddings.Zip(data, (vector, entry) => (vector, entry.UID)))
                        g.FillEllipse(colors[int.Parse(uid)], vector.X * width, vector.Y * height, 5, 5);
                }
                bitmap.Save("Output-Color.png");
            }

            Console.WriteLine("Generated visualisation images");
            Console.WriteLine("Press [Enter] to terminuate..");
            Console.ReadLine();
        }
    }

    [MessagePackObject]
    public sealed class LabelledVector
    {
        [Key(0)] public string UID;
        [Key(1)] public float[] Vector;
    }
}