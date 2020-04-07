using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace DigitRecognizer
{
    public class DigitInputData2
    {
        [ColumnName("PixelValues")]
        [VectorType(49)]
        public float[] PixelValues { get; set; }

        public float Number { get; set; }
    }
}
