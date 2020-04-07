using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Hello_ML.NET_World
{
    internal class HouseRegression1
    {
        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        static void Main()
        {
            MLContext mlContext = new MLContext();

            // 1. Uvezi ili stvori trening podatke
            HouseData[] houseData = {
                new HouseData { Size = 1.1F, Price = 1.2F },
                new HouseData { Size = 1.9F, Price = 2.3F },
                new HouseData { Size = 2.8F, Price = 3.0F },
                new HouseData { Size = 3.4F, Price = 3.7F } };

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // 2. Specificiraj pipeline za pripremu podataka i trening
            var pipeline = mlContext
                    .Transforms
                    .Concatenate(
                        outputColumnName:"Features",
                        inputColumnNames: new[] { "Size" })
                    .Append(mlContext.Regression.Trainers.Sdca(
                        labelColumnName: "Price",
                        featureColumnName: "Features",
                        maximumNumberOfIterations: 100));

            // 3. Treniraj model
            var model = pipeline.Fit(trainingData);

            // 4. Testiraj model
            HouseData[] testHouseData =
            {
                new HouseData { Size = 1.1F, Price = 0.98F },
                new HouseData { Size = 1.9F, Price = 2.1F },
                new HouseData { Size = 2.8F, Price = 2.9F },
                new HouseData { Size = 3.4F, Price = 3.6F }
            };

            IDataView testHouseDataView = mlContext.Data.LoadFromEnumerable(testHouseData);
            IDataView testPriceDataView = model.Transform(testHouseDataView);
            var debug = testPriceDataView.Preview();

            var metrics = mlContext.Regression.Evaluate(
                testPriceDataView,
                labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            // 5. Spremi model
            mlContext.Model.Save(model, trainingData.Schema, "model.zip");

            // 6. Radi predikciju
            MakePrediction();
        }

        // 6. Radi predikciju
        static void MakePrediction()
        {
            MLContext mlContext = new MLContext();
            ITransformer trainedModel = mlContext.Model.Load("model.zip", out var modelInputSchema);

            var size = new HouseData { Size = 2.5F };
            Prediction price = mlContext.Model
                .CreatePredictionEngine<HouseData, Prediction>(trainedModel)
                .Predict(size);

            Console.WriteLine($"Predviđena cijena za veličinu: {size.Size * 1000} sq ft = ${price.Price * 100}k");
        }
    }
}