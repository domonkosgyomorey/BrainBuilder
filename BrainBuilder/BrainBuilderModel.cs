using Newtonsoft.Json;
using System.Text.Json;

namespace BrainBuilder
{
    public class BrainBuilderModel
    {
        public string[] Layers
        {
            get; set;
        }
        public double LearningRate
        {
            get; set;
        }
        public string LossFunction
        {
            get; set;
        }

        public string ToJson()
        {
            var modelData = new
            {
                Layers = Layers,
                LearningRate = LearningRate,
                LossFunction = LossFunction
            };

            return JsonConvert.SerializeObject(modelData, Newtonsoft.Json.Formatting.Indented);
        }
    }

}
