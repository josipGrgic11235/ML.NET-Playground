using System;
using System.IO;
using Microsoft.ML;

namespace Hello_ML.NET_World_2
{
    internal class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "data", "taxi-fare-test.csv");

        static void Main()
        {
            MLContext mlContext = new MLContext();

            var model = Train(mlContext, _trainDataPath);
            Evaluate(mlContext, model);
            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                // strings (VendorId, RateCode, and PaymentType) to numbers (VendorIdEncoded, RateCodeEncoded, and PaymentTypeEncoded)
                // OneHotEncodingTransformer -> assigns different numeric key values to the different values in each of the columns
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                // combine all of the feature columns into the Features column 
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                // Pick a learning algorithm
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample1 = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            // One copied directly from the training data
            // vendor_id,   rate_code,  passenger_count,    trip_time_in_secs,  trip_distance,  payment_type,   fare_amount
            // CMT,         1,          1,                  243,                0.6,            CSH,            4.5
            var taxiTripSample2 = new TaxiTrip()
            {
                VendorId = "CMT",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 243,
                TripDistance = 0.6f,
                PaymentType = "CSH",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction1 = predictionEngine.Predict(taxiTripSample1);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction1.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");

            var prediction2 = predictionEngine.Predict(taxiTripSample2);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction2.FareAmount:0.####}, actual fare: 4.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
