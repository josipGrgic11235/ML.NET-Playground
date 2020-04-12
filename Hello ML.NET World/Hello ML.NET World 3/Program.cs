using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.FastTree;

namespace Hello_ML.NET_World_3
{
    class Program
    {
        static void Main()
        {
            (var trainData, var testData, var validationData) = GetData(SineFunction);

            var mlContext = new MLContext();

            var trainingData = mlContext.Data.LoadFromEnumerable(trainData);

            // 2. Specificiraj pipeline za pripremu podataka i trening
            var pipeline = mlContext
                    .Transforms
                    .Concatenate(
                        outputColumnName: "Features",
                        inputColumnNames: new[] { "Input" })
                    .Append(mlContext.Regression.Trainers.FastTreeTweedie(
                        labelColumnName: "Output",
                        featureColumnName: "Features"
                        ));

            // 3. Treniraj model
            var model = pipeline.Fit(trainingData);

            var testInputDataView = mlContext.Data.LoadFromEnumerable(testData);
            var testOutputDataView = model.Transform(testInputDataView);
            var debug = testOutputDataView.Preview();

            var metrics = mlContext.Regression.Evaluate(
                testOutputDataView,
                labelColumnName: "Output");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            var engine = mlContext.Model.CreatePredictionEngine<InputData, Prediction>(model);
            var sum = 0f;
            validationData.OrderBy(x => x.Input).ToList().ForEach(x =>
            {
                var prediction = engine.Predict(x);

                var delta = prediction.Output - x.Output;
                sum += delta * delta;

                Console.WriteLine($"f({x.Input}) = {prediction.Output}; Actual = {x.Output}; Delta = {delta}");
            });

            Console.WriteLine($"RMSD = {Math.Sqrt(sum / validationData.Count())}");
        }

        private static (IEnumerable<InputData>, IEnumerable<InputData>, IEnumerable<InputData>) GetData(Func<float, float> generator)
        {
            var rnd = new Random();

            var data = new List<InputData>();

            for (var x = 1; x < 100000; x++)
            {
                var input = ((float)x) / 1000;
                data.Add(new InputData
                {
                    Input = input,
                    Output = generator(input)
                });
            }

            var randomData = data.OrderBy(x => rnd.Next());

            var trainData = randomData.Take(80000);
            var testData = randomData.Skip(trainData.Count()).Take(20000 - 100);
            var validationData = randomData.Skip(trainData.Count() + testData.Count()).Take(100);

            return (trainData.ToList(), testData.ToList(), validationData.ToList());
        }

        public static float LinearFunction(float x)
        {
            return 3 * x + 15;
        }

        public static float QuadraticFunction(float x)
        {
            return x * x + 4 * x - 10;
        }

        public static float SineFunction(float x)
        {
            return  100 * (float) Math.Sin(x) + 110;
        }
    }

    public class InputData
    {
        public float Input { get; set; }
        public float Output { get; set; }
    }

    public class Prediction
    {
        [ColumnName("Score")]
        public float Output { get; set; }
    }
}
