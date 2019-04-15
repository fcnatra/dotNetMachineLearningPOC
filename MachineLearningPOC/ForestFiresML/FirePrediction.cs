﻿using Microsoft.ML.Data;

namespace MachineLearningPOC.ForestFiresML
{
    // IrisPrediction is the result returned from prediction operations
    public class FirePrediction
    {
        [ColumnName("PredictedLabel")]
        public double PredictedLabels;
    }
}
