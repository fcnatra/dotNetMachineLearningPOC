using Microsoft.ML;

namespace MachineLearningPOC.FlowertTypeML
{
    public class FlowerTypePredictor
    {
        private MLContext dotNetMachineLearningContext;

        public string PredictedLabels { get; set; } = string.Empty;

        public void Predict()
        {
            CreateMachineLearningContext();

            IDataView trainingDataView = LoadHistoricDataToLearnThePatternsFrom();

            IEstimator<ITransformer> pipeLine = SetUpLearningModel();

            ITransformer model = Train(trainingDataView, pipeLine);

            IrisPrediction prediction = PredictBasedOnTheModel(model);

            this.PredictedLabels = prediction.PredictedLabels;
        }

        private IrisPrediction PredictBasedOnTheModel(ITransformer model)
        {
            // Use your model to make a prediction
            // You can change these numbers to test different predictions
            return dotNetMachineLearningContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth  = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth  = 5.1f,
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
            IEstimator<ITransformer> pipeLine = dotNetMachineLearningContext.Transforms.Conversion.MapValueToKey("Label");
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeLine = pipeLine.AppendCacheCheckpoint(dotNetMachineLearningContext);

            //// add a learner
            //// Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            //// Assign numeric values to text in the "Label" column,
            //// because only numbers can be processed during model training.
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"));

            //// Convert the Label back into original text (after converting to number in step 3)
            pipeLine = pipeLine.Append(dotNetMachineLearningContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return pipeLine;
        }

        private IDataView LoadHistoricDataToLearnThePatternsFrom()
        {
            // Make sure the file is copied to the output folder
            IDataView data = dotNetMachineLearningContext
                .Data
                .LoadFromTextFile<IrisData>(path: "FlowerTypeML\\iris-data.txt", hasHeader: false, separatorChar: ',');
            return data;
        }

        private void CreateMachineLearningContext()
        {
            dotNetMachineLearningContext = new MLContext();
        }
    }
}
