using BrainBuilder.Layers;
using MathNet.Numerics.LinearAlgebra;
using System.Text.Json;

namespace BrainBuilder.Serializer
{
    public static class LayerFactory
    {
        public static ILayer FromJson(string json)
        {
            using (var document = JsonDocument.Parse(json))
            {
                var root = document.RootElement;

                if (root.TryGetProperty("Type", out var typeProperty))
                {
                    var type = typeProperty.GetString();

                    switch (type)
                    {
                        case "Dense":
                            return Dense.FromJson(root);
                        case "BatchNormalization":
                            return BatchNormalization.FromJson(root);
                        case "Activation":
                            return Activation.FromJson(root);
                        default:
                            throw new InvalidOperationException("Unknown layer type.");
                    }
                }
                else
                {
                    throw new InvalidOperationException("Layer type not specified in JSON.");
                }
            }
        }
    }
}
