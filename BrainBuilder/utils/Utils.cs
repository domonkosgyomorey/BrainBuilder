using MathNet.Numerics.LinearAlgebra;
using System.Text.Json;

namespace BrainBuilder.utils
{
    public static class Utils
    {
        public static Random random = new Random();

        public class MatrixJsonSerializer
        {
            public static JsonElement ToJson(Matrix<double> matrix, string name)
            {
                using (var stream = new MemoryStream())
                {
                    using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
                    {
                        writer.WriteStartObject();
                        writer.WriteString("Type", "Matrix");
                        writer.WriteNumber("Rows", matrix.RowCount);
                        writer.WriteNumber("Columns", matrix.ColumnCount);

                        writer.WritePropertyName(name);
                        writer.WriteStartArray();
                        for (int i = 0; i < matrix.RowCount; i++)
                        {
                            writer.WriteStartArray();
                            for (int j = 0; j < matrix.ColumnCount; j++)
                            {
                                writer.WriteNumberValue(matrix[i, j]);
                            }
                            writer.WriteEndArray();
                        }
                        writer.WriteEndArray();

                        writer.WriteEndObject();
                        writer.Flush();
                    }

                    return JsonDocument.Parse(stream.ToArray()).RootElement;
                }
            }
        }
    }
}
