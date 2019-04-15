using Microsoft.ML.Data;

namespace MachineLearningPOC.FlowertTypeML
{
    // IrisPrediction is the result returned from prediction operations
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
