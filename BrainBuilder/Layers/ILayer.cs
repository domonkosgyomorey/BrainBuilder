using MathNet.Numerics.LinearAlgebra;
using System.Text.Json;

namespace BrainBuilder.Layers{
    public interface ILayer {

        Vector<double> Feedforward(Vector<double> input);
        Vector<double> Learn(double learning_rate, Vector<double> xi);
        string ToJson();

        extern static ILayer FromJson(JsonElement root);
    }
}
