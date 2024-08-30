using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Text.Json;

namespace BrainBuilder.Layers
{
    public class Dense : ILayer
    {
        private int _inputSize;
        private int _outputSize;
        private Vector<double> _lastInput;
        private Matrix<double> _weight;
        private Vector<double> _bias;

        public Matrix<double> Weight
        {
            get => _weight; 
            set => _weight = value;
        }

        public Vector<double> LastInput
        {
            get => _lastInput;
            set => _lastInput = value;
        }

        public Vector<double> Bias
        {
            get => _bias; 
            set => _bias = value;
        }

        public Dense(int inputSize, int outputSize)
        {
            this._inputSize = inputSize;
            this._outputSize = outputSize;

            this._weight = Matrix<double>.Build.Random(outputSize, inputSize);
            this._bias = Vector<double>.Build.Random(outputSize);
            this._lastInput = Vector<double>.Build.Dense(inputSize);
        }

        public Vector<double> Feedforward(Vector<double> input)
        {
            _lastInput = input.Clone();
            return _weight * input + _bias;
        }

        public Vector<double> Learn(double learningRate, Vector<double> xi)
        {
            var dw = xi.ToColumnMatrix() * _lastInput.ToRowMatrix();
            _weight -= dw * learningRate;
            _bias -= xi * learningRate;
            return _weight.Transpose() * xi;
        }

        public string ToJson()
        {
            using(var stream = new MemoryStream())
            {
                using(var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
                {
                    writer.WriteStartObject();
                    writer.WriteString("Type", "Dense");

                    writer.WritePropertyName("Weights");
                    writer.WriteStartObject();
                    writer.WriteString("Type", "Matrix");
                    writer.WriteNumber("Rows", Weight.RowCount);
                    writer.WriteNumber("Columns", Weight.ColumnCount);
                    writer.WritePropertyName("Data");
                    writer.WriteStartArray();
                    for(int i = 0; i < Weight.RowCount; i++)
                    {
                        writer.WriteStartArray();
                        for(int j = 0; j < Weight.ColumnCount; j++)
                        {
                            writer.WriteNumberValue(Weight[i, j]);
                        }
                        writer.WriteEndArray();
                    }
                    writer.WriteEndArray();
                    writer.WriteEndObject();

                    writer.WritePropertyName("Bias");
                    writer.WriteStartArray();
                    foreach(var value in Bias)
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
            if(root.TryGetProperty("Weights", out var weightsProperty) &&
    root.TryGetProperty("Bias", out var biasProperty))
            {
                var weightsElement = weightsProperty.GetProperty("Data");
                var rows = weightsProperty.GetProperty("Rows").GetInt32();
                var columns = weightsProperty.GetProperty("Columns").GetInt32();
                var weightMatrix = Matrix<double>.Build.Dense(rows, columns, (i, j) => weightsElement[i][j].GetDouble());

                var bias = biasProperty.EnumerateArray().Select(x => x.GetDouble()).ToArray();
                var biasVector = Vector<double>.Build.Dense(bias);

                var denseLayer = new Dense(columns, rows)
                {
                    Weight = weightMatrix,
                    Bias = biasVector
                };

                return denseLayer;
            }
            throw new InvalidOperationException("Dense layer parameters not specified in JSON.");
        }
    }
}
