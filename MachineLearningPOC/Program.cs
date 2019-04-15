
using MachineLearningPOC.FlowertTypeML;
using MachineLearningPOC.ForestFiresML;
using System;

namespace MachineLearningPOC
{
    partial class Program
    {
        static void Main(string[] args)
        {
            var flowerPredictor = new FlowerTypePredictor();
            flowerPredictor.Predict();
            Console.WriteLine($"Predicted flower type is: {flowerPredictor.PredictedLabels}");

            var firePredictor = new FirePredictor();
            firePredictor.Predict();
            Console.WriteLine($"Predicted humidity is: {firePredictor.PredictedLabels}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();

        }
    }
}
