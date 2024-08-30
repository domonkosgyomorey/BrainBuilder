using BrainBuilder.Layers;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Concurrent;
using Newtonsoft.Json;
using BrainBuilder.Serializer;

namespace BrainBuilder
{
    public class BrainBuilder
    {
        private ILayer[] _layers;
        private double _learningRate;
        private Loss _lossFunction;
        private MomentumLearningRateHandler _lrHandler;
        private object _serializedLayers;
        private Action<List<double>, Dictionary<Vector<double>, Vector<double>>, int> _trainFn;

        public enum Optimizer
        {
            SGD,
            None
        }

        public BrainBuilder(ILayer[] layers, Loss.Type lossType, MomentumLearningRateHandler lrHandler)
        {
            _layers = layers;
            _lossFunction = new Loss(lossType);
            _lrHandler = lrHandler;
        }

        public Vector<double> Feedforward(Vector<double> input)
        {
            Vector<double> nextLayerInput = input;
            foreach(var layer in _layers)
            {
                nextLayerInput = layer.Feedforward(nextLayerInput);
            }
            return nextLayerInput;
        }

        public void Train(Optimizer optimizer, int maxEpoch, Dictionary<Vector<double>, Vector<double>> trainingData, int batchSize = 1)
        {
            switch(optimizer)
            {
                case Optimizer.None:
                    _trainFn = Backpropagation;
                    break;
                case Optimizer.SGD:
                    _trainFn = SGD;
                    break;
            }

            for(int epoch = 1; epoch <= maxEpoch; epoch++)
            {
                _learningRate = _lrHandler.GetLearningRate();
                List<double> losses = new List<double>();
                _trainFn(losses, trainingData, batchSize);
                Console.WriteLine($"Epoch {epoch}/{maxEpoch}, Avg Loss: {losses.Average()}, lr: {_learningRate}");
            }
        }

        private void Backpropagation(List<double> losses, Dictionary<Vector<double>, Vector<double>> trainingData, int batch=1)
        {
            foreach((Vector<double> data, Vector<double> label) in trainingData)
            {
                Vector<double> prediction = Feedforward(data);
                double loss = _lossFunction.GetLoss(prediction, label);
                losses.Add(loss);
                Vector<double> gradient = _lossFunction.GetLossDerivative(prediction, label);
                Learn(gradient);
            }
        }

        private void SGD(List<double> losses, Dictionary<Vector<double>, Vector<double>> trainingData, int batchSize)
        {
            Vector<double> overallGradientSum = Vector<double>.Build.Dense(trainingData.First().Value.Count);
            ConcurrentDictionary<Vector<double>, Vector<double>> data = new ConcurrentDictionary<Vector<double>, Vector<double>>(trainingData);
            var batches = GetMiniBatches(data, batchSize).ToList();
            Parallel.ForEach(batches, batch =>
            {
                Vector<double> batchGradientSum = Vector<double>.Build.Dense(batch.First().Value.Count);

                foreach(var entry in batch)
                {
                    Vector<double> data = entry.Key;
                    Vector<double> label = entry.Value;
                    Vector<double> prediction = Feedforward(data);
                    double loss = _lossFunction.GetLoss(prediction, label);
                    lock(losses)
                        losses.Add(loss);

                    Vector<double> gradient = _lossFunction.GetLossDerivative(prediction, label);
                    batchGradientSum += gradient / batch.Count;
                }
                lock(overallGradientSum)
                    overallGradientSum += batchGradientSum;
            });
            Learn(overallGradientSum);
        }

        private void Learn(Vector<double> gradient)
        {
            Vector<double> xi = gradient;
            foreach(ILayer layer in _layers.Reverse())
            {
                xi = layer.Learn(_learningRate, xi);
            }
        }

        private IEnumerable<Dictionary<Vector<double>, Vector<double>>> GetMiniBatches(ConcurrentDictionary<Vector<double>, Vector<double>> data, int batchSize)
        {
            var shuffledData = data.OrderBy(x => Guid.NewGuid()).ToList(); // Random shuffle
            for(int i = 0; i < shuffledData.Count; i += batchSize)
            {
                yield return shuffledData.Skip(i).Take(batchSize).ToDictionary(x => x.Key, x => x.Value);
            }
        }

        public void Save(string filePath)
        {
            var layersData = _layers.Select(layer => layer.ToJson()).ToArray();
            var modelData = new BrainBuilderModel
            {
                Layers = layersData,
                LearningRate = _learningRate,
                LossFunction = _lossFunction.LossType.ToString(),
            };

            var json = JsonConvert.SerializeObject(modelData, Formatting.Indented);
            File.WriteAllText(filePath, json);
        }

        public static BrainBuilder Load(string filePath, Loss.Type lossType, MomentumLearningRateHandler lrHandler)
        {
            var json = File.ReadAllText(filePath);
            var modelData = JsonConvert.DeserializeObject<BrainBuilderModel>(json);

            if(modelData == null)
                throw new InvalidOperationException("Unable to deserialize model data.");

            var layers = modelData.Layers.Select(LayerFactory.FromJson).ToArray();

            return new BrainBuilder(layers, lossType, lrHandler);
        }
    }
}
