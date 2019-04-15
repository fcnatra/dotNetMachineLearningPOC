using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearningPOC.ForestFiresML
{
    public class ForestFireData
    {
        /// <summary>
        /// x-axis spatial coordinate within the Montesinho park map: 1 to 9
        /// </summary>
        [LoadColumn(0)]
        public int X { get; set; }

        /// <summary>
        /// Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
        /// </summary>
        [LoadColumn(1)]
        public int Y { get; set; }

        /// <summary>
        /// month - month of the year: "jan" to "dec" 
        /// </summary>
        [LoadColumn(2)]
        public string Month { get; set; }

        /// <summary>
        /// day - day of the week: "mon" to "sun"
        /// </summary>
        [LoadColumn(3)]
        public string WeekDay { get; set; }

        /// <summary>
        /// FFMC index from the FWI system: 18.7 to 96.20
        /// </summary>
        [LoadColumn(4)]
        public double FFMC { get; set; }

        /// <summary>
        /// DMC index from the FWI system: 1.1 to 291.3 
        /// </summary>
        [LoadColumn(5)]
        public double DMC { get; set; }

        /// <summary>
        /// DC index from the FWI system: 7.9 to 860.6 
        /// </summary>
        [LoadColumn(6)]
        public double DC { get; set; }

        /// <summary>
        /// ISI index from the FWI system: 0.0 to 56.10
        /// </summary>
        [LoadColumn(7)]
        public double ISI { get; set; }

        /// <summary>
        /// temperature in Celsius degrees: 2.2 to 33.30
        /// </summary>
        [LoadColumn(8)]
        public double Temp { get; set; }

        /// <summary>
        /// relative humidity in %: 15.0 to 100
        /// </summary>
        [LoadColumn(9)]
        public double RH { get; set; }

        /// <summary>
        /// wind speed in km/h: 0.40 to 9.40 
        /// </summary>
        [LoadColumn(10)]
        public double Wind { get; set; }

        /// <summary>
        /// outside rain in mm/m2 : 0.0 to 6.4 
        /// </summary>
        [LoadColumn(11)]
        public double Rain { get; set; }

        /// <summary>
        /// the burned area of the forest (in ha): 0.00 to 1090.84
        /// (this output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform)
        /// </summary>
        [LoadColumn(12)]
        public double Area { get; set; }
    }
}
