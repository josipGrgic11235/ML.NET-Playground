using System;
using System.Collections.Generic;
using System.IO;

namespace MNISTLoader
{
    public class MNISTDigitLoader
    {
        public static MNISTDigitData LoadDigits(string imageIdxFilePath, string imageLabelIdxPath)
        {
            if (!File.Exists(imageIdxFilePath) || !File.Exists(imageLabelIdxPath))
            {
                throw new FileNotFoundException();
            }

            var digitData = new MNISTDigitData();
            using (var imageReader = new BinaryReader(File.Open(imageIdxFilePath, FileMode.Open)))
            {
                using var labelReader = new BinaryReader(File.Open(imageLabelIdxPath, FileMode.Open));
                
                var imagesFileMagicNumber = imageReader.ReadInt32BE();
                var labelsFileMagicNumber = labelReader.ReadInt32BE();

                var numberOfImages = imageReader.ReadInt32BE();
                var numberOfLabels = labelReader.ReadInt32BE();

                if (numberOfLabels != numberOfImages)
                {
                    throw new Exception("Nums of labels and images are different!");
                }

                var singleImageHeight = imageReader.ReadInt32BE();
                var singleImageWidth = imageReader.ReadInt32BE();

                digitData.ImageCount = numberOfImages;
                digitData.SingleImageHeight = singleImageHeight;
                digitData.SingleImageWidth = singleImageWidth;

                var pixelsPerImageCount = singleImageWidth * singleImageHeight;

                for (var i = 0; i < numberOfImages; i++)
                {
                    var imageBytes = imageReader.ReadBytes(pixelsPerImageCount);
                    var imageLabel = labelReader.ReadByte();
                    digitData.Images.Add(new Image
                    {
                        Height = digitData.SingleImageHeight,
                        Width = digitData.SingleImageWidth,
                        Pixels = imageBytes,
                        Label = imageLabel
                    });
                }
            }

            return digitData;
        }
    }

    public static class BinaryReaderExtensions
    {
        public  static int ReadInt32BE(this BinaryReader reader)
        {
            var data = reader.ReadBytes(4);
            Array.Reverse(data);
            return BitConverter.ToInt32(data, 0);
        }
    }

    public class MNISTDigitData
    {
        public int ImageCount { get; set; }
        public int SingleImageHeight { get; set; }
        public int SingleImageWidth { get; set; }

        /// <summary>
        /// Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        /// </summary>
        public List<Image> Images { get; set; }

        public MNISTDigitData()
        {
            Images = new List<Image>();
        }
    }

    public class Image
    {
        public byte[] Pixels { get; set; }
        public byte Label { get; set; }

        public int Width { get; set; }
        public int Height { get; set; }
    }
}
