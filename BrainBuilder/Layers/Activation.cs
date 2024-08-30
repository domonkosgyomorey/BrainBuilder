using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Text.Json;

namespace BrainBuilder.Layers
{
    public class Activation:ILayer
    {
        private Vector<double> _lastInput;
        private Func<double, double> _activation;
        private Func<double, double> _activationDerivative;
        private static double MAX_VAL = 1E6;

        private double _leakyCoef = 0.1;
        private bool _softmax = false;
        private Type activationType;

        public enum Type
        {
            Sigmoid,
            ReLU,
            TanH,
            LeakyReLU,
            ELU,
            Softmax
        };

        public Activation(Activation.Type type = Type.Sigmoid)
        {
            activationType = type;
            switch(type)
            {
                case Type.Sigmoid:
                    _activation = Sigmoid;
                    _activationDerivative = SigmoidD;
                    break;
                case Type.ReLU:
                    _activation = Relu;
                    _activationDerivative = ReluD;
                    break;
                case Type.TanH:
                    _activation = Tanh;
                    _activationDerivative = TanhD;
                    break;
                case Type.LeakyReLU:
                    _activation = LeakyRelu;
                    _activationDerivative = LeakyReluD;
                    break;
                case Type.ELU:
                    _activation = Elu;
                    _activationDerivative = EluD;
                    break;
                case Type.Softmax:
                    _softmax = true;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }

        public Vector<double> Feedforward(Vector<double> input)
        {
            _lastInput = input.Clone();
            Vector<double> output;
            if(_softmax)
            {
                output = Softmax(input);
            } else
            {
                output = input.Map(_activation);
            }
            for (int i = 0; i < output.Count; i++)
            {
                output[i] = Math.Clamp(output[i], -MAX_VAL, MAX_VAL);
            }
            return output;
        }

        public Vector<double> Learn(double learningRate, Vector<double> xi)
        {
            if(_softmax)
            {
                return xi - Softmax(_lastInput) * xi.Sum();
            }
            return xi.PointwiseMultiply(_lastInput.Map(_activationDerivative));
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidD(double x) => Sigmoid(x) * (1.0 - Sigmoid(x));
        private double Relu(double x) => Math.Max(0, x);
        private double ReluD(double x) => x > 0 ? 1 : 0;
        private double Tanh(double x) => Math.Tanh(x);
        private double TanhD(double x) => 1.0 - Math.Pow(Tanh(x), 2);
        private double LeakyRelu(double x) => x > 0 ? x : _leakyCoef * x;
        private double LeakyReluD(double x) => x > 0 ? 1 : _leakyCoef;

        private double Elu(double x) => x <= 0 ? 2 * (Math.Exp(x) - 1) : x;
        private double EluD(double x) => x <= 0 ? 2 * Math.Exp(x) : 1;
        private Vector<double> Softmax(Vector<double> xs)
        {
            Vector<double> result = Vector<double>.Build.Dense(xs.Count);
            for(int i = 0; i < xs.Count; i++)
            {
                result[i] = Math.Exp(xs[i]) / xs.PointwiseExp().Sum();
            }
            return result;
        }

        public string ToJson()
        {
            using(var stream = new MemoryStream())
            {
                using(var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
                {
                    writer.WriteStartObject();
                    writer.WriteString("Type", "Activation");
                    writer.WriteString("ActivationFunction", activationType.ToString());
                    writer.WriteEndObject();
                    writer.Flush();
                }
                return Encoding.UTF8.GetString(stream.ToArray());
            }
        }

        public static ILayer FromJson(JsonElement root)
        {
            if(root.TryGetProperty("ActivationFunction", out var activationProperty))
            {
                var activationType = Enum.Parse<Type>(activationProperty.GetString(), true);

                return new Activation(activationType);
            }
            throw new InvalidOperationException("Activation layer type not specified in JSON.");
        }
    }
}
