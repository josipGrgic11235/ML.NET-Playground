
using Microsoft.ML.Data;

namespace DigitRecognizer
{
    public class DigitInputData
    {
        [ColumnName("PixelValues")]
        [VectorType(28 * 28)]
        public float[] PixelValues { get; set; }

        public float Number { get; set; }
    }
}
