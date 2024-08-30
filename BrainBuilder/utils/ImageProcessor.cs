using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BrainBuilder.utils
{
    public static class ImageProcessor
    {
        public static double[] LoadAndProcessImage(string filePath)
        {
            using (Image<L8> image = Image.Load<L8>(filePath)) // L8 is for 8-bit grayscale
            {
                int width = image.Width;
                int height = image.Height;
                double[] pixelData = new double[width * height];

                int index = 0;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        byte pixelValue = image[x, y].PackedValue;
                        pixelData[index++] = pixelValue / 255.0;
                    }
                }

                return pixelData;
            }
        }
    }
}
