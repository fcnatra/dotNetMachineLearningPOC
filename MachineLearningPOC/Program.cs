
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
            Console.WriteLine($"Predicted flower type is: {flowerPredictor.PredictedLabels}" +
                $"{Environment.NewLine}\tusing this entry data: {flowerPredictor.EntryData.ToString()}");

            var firePredictor = new FirePredictor();
            firePredictor.Predict();
            Console.WriteLine($"Predicted week day is: {firePredictor.PredictedWeekDays}" +
                $"{Environment.NewLine}\tusing this entry data: {firePredictor.EntryData.ToString()}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();

        }
    }
}
