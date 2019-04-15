
using System;

namespace MachineLearningPOC
{
    partial class Program
    {
        static void Main(string[] args)
        {
            var predictor = new FlowerTypePredictor();
            predictor.Predict();

            Console.WriteLine($"Predicted flower type is: {predictor.PredictedLabels}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();

        }
    }
}
