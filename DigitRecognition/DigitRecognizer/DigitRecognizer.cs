using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System.IO;

namespace DigitRecognizer
{
    public class DigitRecognizer
    {
        private static string ModelSubPath = "C:\\josip.grgic\\DigitRecognition\\Console\\MLModels";
        private static string ModelPath = $"{ModelSubPath}\\Model.zip";

        public static void Train(IEnumerable<DigitInputData> trainImages, IEnumerable<DigitInputData> testImages)
        {
            /*var mlContext = new MLContext();
            var trainData = mlContext.Data.LoadFromEnumerable(trainImages);
            var testData = mlContext.Data.LoadFromEnumerable(testImages);

            var dataProcessPipeline = mlContext
                .Transforms
                .Conversion
                .MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.Concatenate("Features", nameof(DigitInputData.PixelValues)));

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy();
            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));

            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainData);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number");

            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            for (var i = 0; i < 10; i++)
            {
                PredictDigit(trainImages.ToList()[i]);
            }*/

            var mlContext = new MLContext();
            var trainData = mlContext.Data.LoadFromEnumerable(trainImages.Skip(10));
            var testData = mlContext.Data.LoadFromEnumerable(testImages);

            /*var dataProcessPipeline = mlContext
                .Transforms.Conversion
                .MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.Concatenate("Features", "PixelValues"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));
                */

            var dataProcessPipeline = mlContext
                .Transforms.Conversion
                .MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.NormalizeMinMax("PixelValues"));

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "PixelValues");

            var trainingPipeline = dataProcessPipeline
                .Append(trainer);

            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainData);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            _ = Directory.CreateDirectory(ModelSubPath);
            mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            PredictDigit(GetZero());
            for (var i = 0; i < 10; i++)
            {
                PredictDigit(trainImages.ToList()[i]);
            }
        }

        public static void Train2(IEnumerable<DigitInputData> trainImages, IEnumerable<DigitInputData> testImages)
        {
            var mlContext = new MLContext();
            

            var mappedTrainImages = trainImages.Select(MapToAlternateInput);
            var mappedTestImages = testImages.Select(MapToAlternateInput);

            var trainData = mlContext.Data.LoadFromEnumerable(mappedTrainImages.Skip(10));
            var testData = mlContext.Data.LoadFromEnumerable(mappedTestImages);

            var dataProcessPipeline = mlContext
                .Transforms.Conversion
                .MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue);

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "PixelValues");

            var trainingPipeline = dataProcessPipeline
                .Append(trainer);

            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainData);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            for (var i = 0; i < 10; i++)
            {
                PredictDigit2(trainImages.ToList()[i]);
            }
        }

        public static DigitInputData2 MapToAlternateInput(DigitInputData data)
        {
            var array = new float[49];

            for (var i = 0; i < 28; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    if (data.PixelValues[i * 28 + j] > 0)
                    {
                        var iCoord = i / 4;
                        var jCoord = j / 4;

                        array[iCoord * 7 + jCoord]++;
                    }
                }
            }

            return new DigitInputData2
            {
                PixelValues = array,
                Number = data.Number
            };
        }

        public static int PredictDigit(DigitInputData digit)
        {
            PrintDigit(digit);
            var mlContext = new MLContext();
            var trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<DigitInputData, DigitOutputData>(trainedModel);

            var prediction = predEngine.Predict(digit);
            var predictedDigit = 0;
            var maxProbability = (float) 0;

            for (var i = 0; i < prediction.Score.Length; i++)
            {
                if (!(prediction.Score[i] > maxProbability))
                {
                    continue;
                }

                maxProbability = prediction.Score[i];
                predictedDigit = i;
            }

            Console.WriteLine($"Actual: {digit.Number}     Predicted probability:");
            Console.WriteLine($"                                           zero:  {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           one :  {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           two:   {prediction.Score[2]:0.####}");
            Console.WriteLine($"                                           three: {prediction.Score[3]:0.####}");
            Console.WriteLine($"                                           four:  {prediction.Score[4]:0.####}");
            Console.WriteLine($"                                           five:  {prediction.Score[5]:0.####}");
            Console.WriteLine($"                                           six:   {prediction.Score[6]:0.####}");
            Console.WriteLine($"                                           seven: {prediction.Score[7]:0.####}");
            Console.WriteLine($"                                           eight: {prediction.Score[8]:0.####}");
            Console.WriteLine($"                                           nine:  {prediction.Score[9]:0.####}");
            Console.WriteLine($"Prediction: {predictedDigit}");
            Console.WriteLine();

            return predictedDigit;
        }

        public static int PredictDigit2(DigitInputData digit)
        {
            PrintDigit(digit);
            var mlContext = new MLContext();
            var trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<DigitInputData2, DigitOutputData>(trainedModel);

            var mapped = MapToAlternateInput(GetZero());
            var prediction = predEngine.Predict(mapped);
            var predictedDigit = 0;
            var maxProbability = (float)0;

            for (var i = 0; i < prediction.Score.Length; i++)
            {
                if (!(prediction.Score[i] > maxProbability))
                {
                    continue;
                }

                maxProbability = prediction.Score[i];
                predictedDigit = i;
            }

            Console.WriteLine($"Actual: {digit.Number}     Predicted probability:");
            Console.WriteLine($"                                           zero:  {prediction.Score[0]:0.####}");
            Console.WriteLine($"                                           one :  {prediction.Score[1]:0.####}");
            Console.WriteLine($"                                           two:   {prediction.Score[2]:0.####}");
            Console.WriteLine($"                                           three: {prediction.Score[3]:0.####}");
            Console.WriteLine($"                                           four:  {prediction.Score[4]:0.####}");
            Console.WriteLine($"                                           five:  {prediction.Score[5]:0.####}");
            Console.WriteLine($"                                           six:   {prediction.Score[6]:0.####}");
            Console.WriteLine($"                                           seven: {prediction.Score[7]:0.####}");
            Console.WriteLine($"                                           eight: {prediction.Score[8]:0.####}");
            Console.WriteLine($"                                           nine:  {prediction.Score[9]:0.####}");
            Console.WriteLine($"Prediction: {predictedDigit}");
            Console.WriteLine();

            return predictedDigit;
        }

        private static void PrintDigit(DigitInputData digit)
        {
            Console.WriteLine();
            for (var i = 0; i < 28; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    Console.Write(digit.PixelValues[i*28 + j].ToString().PadLeft(4, ' '));
                }
                Console.WriteLine();
            }
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine("************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine("*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            for (var i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"    LogLoss for class {i} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
            }
            Console.WriteLine("************************************************************");
        }

        public static DigitInputData GetZero()
        {
            var rawData = @"0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0  48  87  87  87  87  87  48   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0 128 192 192 192 192 192 134 187  80   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0 223 239 247 192 192 192 195 225 249 176  96  48   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0 232 240 242 135 135 135 160 239 243 247 255 171  48   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0 241 241 227   0   0   0   0   8 102 245 255 254 242  48   0   0   0   0   0   0   
0   0   0   0   0   0   0 223 241 241  32   0   0   0   0   0   0 102 218 252 248 240 171  40   0   0   0   0   
0   0   0   0   0   0   0 227 236 236   0   0   0   0   0   0   0   0  80 150 182 229 245  97   0   0   0   0   
0   0   0   0   0   0   0 227 227 227   0   0   0   0   0   0   0   0   0   0 116 253 243 239   0   0   0   0   
0   0   0   0   0   0   0 255 255 255   0   0   0   0   0   0   0   0   0   0   0  39 241 241 223   0   0   0   
0   0   0   0   0   0   0 227 227 227   0   0   0   0   0   0   0   0   0   0   0   0 236 236 227   0   0   0   
0   0   0   0   0   0   0 227 227 227   0   0   0   0   0   0   0   0   0   0   0   0 227 227 227   0   0   0   
0   0   0   0   0   0   0 255 255 255   0   0   0   0   0   0   0   0   0   0   0   0 227 227 227   0   0   0   
0   0   0   0   0   0   0 227 227 227   0   0   0   0   0   0   0   0   0   0   0   0 227 227 227   0   0   0   
0   0   0   0   0   0   0 227 227 227   0   0   0   0   0   0   0   0   0   0   0   0 229 227 227   0   0   0   
0   0   0   0   0   0   0 227 232 232   0   0   0   0   0   0   0   0   0   0   4 128 253 234 227   0   0   0   
0   0   0   0   0   0   0 227 241 244  87  48   0   0   0   0   0   0   0   0   0 191 255 199  32   0   0   0   
0   0   0   0   0   0   0  32 241 248 247 128   0   0   0   0   0   0   0  24 245 233 255 219  20   0   0   0   
0   0   0   0   0   0   0   0 102 179 248 240 246 121  87  96  87  87  87 121 238 220 113   4   0   0   0   0   
0   0   0   0   0   0   0   0   0  80 155 219 250 255 192 255 192 192 192 255 242 251  76   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0  52 231 158 255 192 255 192 192 192 255 144  39   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   8  80 159 135 159 135 135 135 159  80   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   
0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0";

            var splited = rawData.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            return new DigitInputData
            {
                PixelValues = splited.Select(x => (float) int.Parse(x)).ToArray()
            };
        }

        public static DigitInputData GetFive()
        {
            var rawData = @"0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0
   0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0
   0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82  82  56  39   0   0   0   0   0
   0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201  78   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0";

            var splited = rawData.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            return new DigitInputData
            {
                PixelValues = splited.Select(x => (float)int.Parse(x)).ToArray()
            };
        }
    }
}
