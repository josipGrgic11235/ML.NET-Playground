using System;
using System.IO;
using System.Linq;
using DigitRecognizer;
using MNISTLoader;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Console
{
    class Program
    {
        #region static path strings
        private static string BaseDatasetsRelativePath = @"../../../MNIST Database/data";
        private static string TrainImageDataRealtivePath = $"{BaseDatasetsRelativePath}/train-images.idx3-ubyte";
        private static string TrainLabelDataRealtivePath = $"{BaseDatasetsRelativePath}/train-labels.idx1-ubyte";

        private static string TestImageDataRealtivePath = $"{BaseDatasetsRelativePath}/t10k-images.idx3-ubyte";
        private static string TestLabelDataRealtivePath = $"{BaseDatasetsRelativePath}/t10k-labels.idx1-ubyte";

        private static string TrainImageDataPath = GetAbsolutePath(TrainImageDataRealtivePath);
        private static string TrainLabelDataPath = GetAbsolutePath(TrainLabelDataRealtivePath);

        private static string TestImageDataPath = GetAbsolutePath(TestImageDataRealtivePath);
        private static string TestLabelDataPath = GetAbsolutePath(TestLabelDataRealtivePath);
        #endregion

        static void Main(string[] args)
        {
            Train();
        }

        public static void Train()
        {
            var trainImageData = MNISTDigitLoader.LoadDigits(TrainImageDataPath, TrainLabelDataPath);
            var testImageData = MNISTDigitLoader.LoadDigits(TestImageDataPath, TestLabelDataPath);

            var trainImages = trainImageData.Images.Select(img => new DigitInputData
            {
                PixelValues = ToFloat(img.Pixels),
                Number = img.Label
            });

            var testImages = testImageData.Images.Select(img => new DigitInputData
            {
                PixelValues = ToFloat(img.Pixels),
                Number = img.Label
            });
            
            DigitRecognizer.DigitRecognizer.Train(trainImages, testImages);
        }

        #region Helpers
        private static float[] ToFloat(byte[] bytes)
        {
            return bytes.Select(b => (float)BitConverter.ToInt32(new byte[] {b, 0, 0, 0}, 0)).ToArray();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
        #endregion
    }
}
