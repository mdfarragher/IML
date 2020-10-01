using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageDetector
{
    // The application class
    class Program
    {
        // A data class that hold one image data record
        public class ImageNetData
        {
            [LoadColumn(0)] public string ImagePath;
            [LoadColumn(1)] public string Label;

            // Load the contents of a TSV file as an object sequence representing images and labels
            public static IEnumerable<ImageNetData> ReadFromCsv(string file)
            {
                return File.ReadAllLines(file)
                    .Select(x => x.Split('\t'))
                    .Select(x => new ImageNetData 
                    { 
                        ImagePath = x[0], 
                        Label = x[1] 
                    });
            }
        }

        // A prediction class that holds only a model prediction.
        public class ImageNetPrediction
        {
            [ColumnName("softmax2")]
            public float[] PredictedLabels;
        }

        // The main application entry point.
        static void Main(string[] args)
        {
            // create a machine learning context
            var mlContext = new MLContext();

            // load the TSV file with image names and corresponding labels
            var data = mlContext.Data.LoadFromTextFile<ImageNetData>("images/tags.tsv", hasHeader: true);

            // set up a learning pipeline
            var pipeline = mlContext.Transforms
            
                // step 1: load the images
                .LoadImages(
                    outputColumnName: "input", 
                    imageFolder: "images", 
                    inputColumnName: nameof(ImageNetData.ImagePath))

                // step 2: resize the images to 224x224
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "input", 
                    imageWidth: 224, 
                    imageHeight: 224, 
                    inputColumnName: "input"))

                // step 3: extract pixels in a format the TF model can understand
                // these interleave and offset values are identical to the images the model was trained on
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input", 
                    interleavePixelColors: true, 
                    offsetImage: 117))

                // step 4: load the TensorFlow model
                .Append(mlContext.Model.LoadTensorFlowModel("models/tensorflow_inception_graph.pb")

                // step 5: score the images using the TF model
                .ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2" },
                    inputColumnNames: new[] { "input" }, 
                    addBatchDimensionInput:true));
                        
            // train the model on the data file (does nothing)
            var model = pipeline.Fit(data);

            // create a prediction engine
            var engine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            // load all 1,000 ImageNet labels
            var labels = File.ReadAllLines("models/imagenet_comp_graph_label_strings.txt");

            // load the tags.tsv file
            var images = ImageNetData.ReadFromCsv("images/tags.tsv");

            // predict what is in each image
            foreach (var image in images)
            {
                Console.Write($"  [{image.ImagePath}]: ");
                var prediction = engine.Predict(image).PredictedLabels;

                // find the best prediction
                var i = 0;
                var best = (from p in prediction 
                            select new { Index = i++, Prediction = p }).OrderByDescending(p => p.Prediction).First();
                var predictedLabel = labels[best.Index];

                // show the corresponding label
                Console.WriteLine($"{predictedLabel} {(predictedLabel != image.Label ? "***WRONG***" : "")}");
            }
        }
    }
}
