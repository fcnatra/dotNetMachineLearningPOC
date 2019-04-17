using Microsoft.ML.Data;
using System;

namespace MachineLearningPOC.FlowertTypeML
{
    // STEP 1: Define your data structures
    // IrisData is used to provide training data, and as
    // input for prediction operations
    // - First 4 properties are inputs/features used to predict the label
    // - Label is what you are predicting, and is only set when training
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string Label;
        public override string ToString()
        {
            return $"{Environment.NewLine}\tSepalLength:\t{SepalLength}" +
                $"{Environment.NewLine}\tSepalWidth:\t{SepalWidth}" +
                $"{Environment.NewLine}\tPetalLength:\t{PetalLength}" +
                $"{Environment.NewLine}\tPetalWidth:\t{PetalWidth}";
        }
    }
}
