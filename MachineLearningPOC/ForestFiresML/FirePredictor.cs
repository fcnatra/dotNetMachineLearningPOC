using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearningPOC.ForestFiresML
{
    public class FirePredictor
    {
        private MLContext dotNetMachineLearningContext;

        public string PredictedWeekDays { get; set; } = string.Empty;
        public ForestFireData EntryData { get; set; }

        public void Predict()
        {
            CreateMachineLearningContext();
            IDataView trainingDataView = LoadHistoricDataToLearnThePatternsFrom();
            IEstimator<ITransformer> pipeLine = SetUpLearningModel();
            ITransformer model = Train(trainingDataView, pipeLine);
            FirePrediction prediction = PredictBasedOnTheModel(model);

            this.PredictedWeekDays = prediction.PredictedWeekDays;
        }

        private FirePrediction PredictBasedOnTheModel(ITransformer model)
        {
            // Use your model to make a prediction
            // You can change these numbers to test different predictions
            this.EntryData = new ForestFireData()
            {
                X = 7,
                Y = 4,
                Month = "oct",
                FFMC = 90.6d,
                DMC = 35.4d,
                DC = 669.1d,
                ISI = 6.7d,
                Temp = 18f,
                RH = 33f,
                Wind = 0.9d,
                Rain = 0d,
                Area = 0d
            };
            return dotNetMachineLearningContext
                .Model
                .CreatePredictionEngine<ForestFireData, FirePrediction>(model)
                .Predict(this.EntryData);
        }

        private static ITransformer Train(IDataView trainingDataView, IEstimator<ITransformer> pipeLine)
        {
            // Train your model based on the data set
            return pipeLine.Fit(trainingDataView);
        }

        private IEstimator<ITransformer> SetUpLearningModel()
        {
            // transform your data
            IEstimator<ITransformer> pipeLine = dotNetMachineLearningContext.Transforms.Conversion.MapValueToKey("WeekDay");
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.Transforms.Concatenate("Features", "Temp", "RH"));

            // add a learner
            pipeLine = pipeLine.AppendCacheCheckpoint(dotNetMachineLearningContext);

            // Add a learning algorithm to the pipeline.
            // Assign numeric values to text in the labeled column,
            // because only numbers can be processed during model training.
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "WeekDay", featureColumnName: "Features"));

            // Convert the Label back into original text (after converting to number in step 3)
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabel",
                inputColumnName: "WeekDay"));

            return pipeLine;
        }

        private IDataView LoadHistoricDataToLearnThePatternsFrom()
        {
            // Make sure the file is copied to the output folder
            // IDataViews are lazy, so no actual loading happens here, just schema validation
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
