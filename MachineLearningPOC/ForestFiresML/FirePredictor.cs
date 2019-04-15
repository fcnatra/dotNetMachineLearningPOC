using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearningPOC.ForestFiresML
{
    public class FirePredictor
    {
        private MLContext dotNetMachineLearningContext;

        public double PredictedLabels { get; set; } = -1;

        public void Predict()
        {
            CreateMachineLearningContext();
            IDataView trainingDataView = LoadHistoricDataToLearnThePatternsFrom();
            IEstimator<ITransformer> pipeLine = SetUpLearningModel();
            ITransformer model = Train(trainingDataView, pipeLine);
            FirePrediction prediction = PredictBasedOnTheModel(model);

            this.PredictedLabels = prediction.PredictedLabels;

        }

        private FirePrediction PredictBasedOnTheModel(ITransformer model)
        {
            // Use your model to make a prediction
            // You can change these numbers to test different predictions
            return dotNetMachineLearningContext.Model.CreatePredictionEngine<ForestFireData, FirePrediction>(model).Predict(
                new ForestFireData()
                {
                    Month = "jun",
                    WeekDay = "mon"
                });
        }

        private static ITransformer Train(IDataView trainingDataView, IEstimator<ITransformer> pipeLine)
        {
            // Train your model based on the data set
            return pipeLine.Fit(trainingDataView);
        }

        private IEstimator<ITransformer> SetUpLearningModel()
        {
            // transform your data
            IEstimator<ITransformer> pipeLine = dotNetMachineLearningContext.Transforms.Conversion.MapValueToKey("RH");
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.Transforms.Concatenate("Features", "Month", "WeekDay"));

            // add a learner
            pipeLine = pipeLine.AppendCacheCheckpoint(dotNetMachineLearningContext);

            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Assign numeric values to text in the "Label" column,
            // because only numbers can be processed during model training.
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "WeekDay", featureColumnName: "RH"));

            // Convert the Label back into original text (after converting to number in step 3)
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return pipeLine;
        }

        private IDataView LoadHistoricDataToLearnThePatternsFrom()
        {
            // Make sure the file is copied to the output folder
            IDataView data = dotNetMachineLearningContext
                .Data
                .LoadFromTextFile<ForestFireData>(path: "ForestFiresML\\forestfires.csv", hasHeader: false, separatorChar: ',');
            return data;
        }

        private void CreateMachineLearningContext()
        {
            dotNetMachineLearningContext = new MLContext();
        }
    }
}
