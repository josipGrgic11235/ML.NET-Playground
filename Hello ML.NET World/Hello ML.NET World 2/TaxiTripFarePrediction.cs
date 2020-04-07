using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Hello_ML.NET_World_2
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
