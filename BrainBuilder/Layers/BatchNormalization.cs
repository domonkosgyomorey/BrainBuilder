using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Text.Json;

namespace BrainBuilder.Layers
{
    public class BatchNormalization:ILayer
    {
        private readonly int _inputSize;
        private Vector<double> _gamma;
        private Vector<double> _beta;

        private Vector<double> _runningMean;
        private Vector<double> _runningVariance;
        private double _momentum = 0.9;
        private double _epsilon = 1e-5;

        private Vector<double>? _centeredInput;
        private Vector<double>? _standardizedInput;

        public int InputSize => _inputSize;

        public Vector<double> Gamma
        {
            get => _gamma;
            set => _gamma = value;
        }
        public Vector<double> Beta
        {
            get => _beta;
            set => _beta = value;
        }
        public Vector<double> RunningMean
        {
            get => _runningMean;
            set => _runningMean = value;
        }
        public Vector<double> RunningVariance
        {
            get => _runningVariance;
            set => _runningVariance = value;
        }
        public double Momentum
        {
            get => _momentum;
            set => _momentum = value;
        }
        public double Epsilon
        {
            get => _epsilon;
            set => _epsilon = value;
        }
        public Vector<double>? CenteredInput
        {
            get => _centeredInput;
            set => _centeredInput = value;
        }
        public Vector<double>? StandardizedInput
        {
            get => _standardizedInput;
            set => _standardizedInput = value;
        }

        public BatchNormalization(int inputSize)
        {
            _inputSize = inputSize;
            _gamma = Vector<double>.Build.Dense(inputSize, 1.0);
            _beta = Vector<double>.Build.Dense(inputSize, 0.0);

            _runningMean = Vector<double>.Build.Dense(inputSize, 0.0);
            _runningVariance = Vector<double>.Build.Dense(inputSize, 1.0);
        }

        public Vector<double> Feedforward(Vector<double> input)
        {
            // Calculate the mean and variance for the current batch
            double batchMean = input.Average();
            Vector<double> meanVec = Vector<double>.Build.Dense(_inputSize, batchMean);

            Vector<double> centeredInput = input.Subtract(meanVec);
            Vector<double> varianceVec = centeredInput.PointwisePower(2);
            double batchVariance = varianceVec.Average();

            // Normalize the input
            _centeredInput = centeredInput;
            _standardizedInput = centeredInput.PointwiseDivide(Vector<double>.Build.Dense(_inputSize, Math.Sqrt(batchVariance + _epsilon)));

            // Apply the learned scale (gamma) and shift (beta)
            Vector<double> output = _gamma.PointwiseMultiply(_standardizedInput).Add(_beta);

            // Update running mean and variance for inference
            _runningMean = _runningMean.Multiply(_momentum).Add(meanVec.Multiply(1 - _momentum));
            _runningVariance = _runningVariance.Multiply(_momentum).Add(Vector<double>.Build.Dense(_inputSize, batchVariance).Multiply(1 - _momentum));

            return output;
        }

        public Vector<double> Learn(double learningRate, Vector<double> xi)
        {
            Vector<double> dGamma = xi.PointwiseMultiply(_standardizedInput);
            Vector<double> dBeta = xi;

            _gamma = _gamma.Subtract(dGamma.Multiply(learningRate));
            _beta = _beta.Subtract(dBeta.Multiply(learningRate));

            Vector<double> dxhat = xi.PointwiseMultiply(_gamma);

            double batchVariance = _centeredInput.PointwisePower(2).Average();
            Vector<double> varianceVec = _centeredInput.PointwisePower(2);
            double varianceSum = varianceVec.Sum();

            Vector<double> dvar = dxhat.PointwiseMultiply(_centeredInput)
                .PointwiseDivide(Vector<double>.Build.Dense(_inputSize, 2 * (Math.Pow(batchVariance, 3) + _epsilon)))
                .Multiply(-0.5);

            Vector<double> dmean = dxhat.PointwiseDivide(Vector<double>.Build.Dense(_inputSize, Math.Sqrt(batchVariance + _epsilon)))
                .Negate()
                .Add(dvar.Multiply(-2.0 / _inputSize).PointwiseMultiply(_centeredInput));

            Vector<double> dx = dxhat.PointwiseDivide(Vector<double>.Build.Dense(_inputSize, Math.Sqrt(batchVariance + _epsilon)))
                .Add(dvar.Multiply(2.0 / _inputSize).PointwiseMultiply(_centeredInput))
                .Add(dmean.Multiply(1.0 / _inputSize));

            return dx;
        }

        public string ToJson()
        {
            using(var stream = new MemoryStream())
            {
                using(var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
                {
                    writer.WriteStartObject();
                    writer.WriteString("Type", "BatchNormalization");

                    writer.WritePropertyName("Gamma");
                    writer.WriteStartArray();
                    foreach(var value in Gamma)
                    {
                        writer.WriteNumberValue(value);
                    }
                    writer.WriteEndArray();

                    writer.WritePropertyName("Beta");
                    writer.WriteStartArray();
                    foreach(var value in Beta)
                    {
                        writer.WriteNumberValue(value);
                    }
                    writer.WriteEndArray();

                    writer.WriteEndObject();
                    writer.Flush();
                }

                return Encoding.UTF8.GetString(stream.ToArray());
            }
        }

        public static ILayer FromJson(JsonElement root)
        {
            if(root.TryGetProperty("Gamma", out var gammaProperty) &&
                root.TryGetProperty("Beta", out var betaProperty))
            {
                var gamma = gammaProperty.EnumerateArray().Select(x => x.GetDouble()).ToArray();
                var beta = betaProperty.EnumerateArray().Select(x => x.GetDouble()).ToArray();

                var inputSize = gamma.Length;

                return new BatchNormalization(inputSize)
                {
                    Gamma = Vector<double>.Build.Dense(gamma),
                    Beta = Vector<double>.Build.Dense(beta),
                    RunningMean = Vector<double>.Build.Dense(inputSize, 0.0),
                    RunningVariance = Vector<double>.Build.Dense(inputSize, 1.0)
                };
            }
            throw new InvalidOperationException("BatchNormalization layer parameters not specified in JSON.");
        }
    }
}
