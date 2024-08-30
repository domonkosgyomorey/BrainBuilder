using BrainBuilder;
using BrainBuilder.Layers;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Concurrent;

namespace BrainBuilder
{
    public class Program
    {
        public void Main(string[] args)
        {
            /*
            var imgBrain = new BrainBuilder([
                    new Dense(784, 200),
                    new Activation(Activation.Type.Sigmoid),
                    new Dense(200, 100),
                    new Activation(Activation.Type.Sigmoid),
                    new Dense(100, 1),
                    new Activation(Activation.Type.Sigmoid)
                ], Loss.Type.MAE, LearningRateHandlerFactory.Create(LrHandlerType.TimeDecay, 0.1, 0.99));

            var imgRecDataset = new Dictionary<Vector<double>, Vector<double>>();

            foreach(var line in File.ReadAllLines("Data/img_recog.txt").ToList()){
                string[] a = line.Split(' ');
                string path = a[0];
                double num = int.Parse(a[1])/10.0;
                for(int i = 0; i < 10; i++)
                {
                    imgRecDataset.TryAdd(Vector<double>.Build.DenseOfArray(ImageProcessor.LoadAndProcessImage("Images/"+path.Replace("x", i.ToString()))), Vector<double>.Build.DenseOfArray([num]));
                }
            }

            imgBrain.Train(1000, imgRecDataset, 2);



            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[0]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[10]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[20]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[30]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[40]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[50]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[60]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[70]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[80]));
            Console.WriteLine(imgBrain.Feedforward(imgRecDataset.Keys.ToList()[90]));

            return 0;
            */

            var brain = new BrainBuilder([
                    new Dense(2, 2),
            new Activation(Activation.Type.Sigmoid),
            new Dense(2, 1),
            new Activation(Activation.Type.Sigmoid)
                ], Loss.Type.CrossEntropy, LearningRateHandlerFactory.Create(LrHandlerType.CosineAnnealing, 0.1));

            var xor_dataset = new Dictionary<Vector<double>, Vector<double>>();
            xor_dataset.TryAdd(Vector<double>.Build.DenseOfArray([0, 0]), Vector<double>.Build.DenseOfArray([0]));
            xor_dataset.TryAdd(Vector<double>.Build.DenseOfArray([0, 1]), Vector<double>.Build.DenseOfArray([1]));
            xor_dataset.TryAdd(Vector<double>.Build.DenseOfArray([1, 0]), Vector<double>.Build.DenseOfArray([1]));
            xor_dataset.TryAdd(Vector<double>.Build.DenseOfArray([1, 1]), Vector<double>.Build.DenseOfArray([0]));

            brain.Train(BrainBuilder.Optimizer.None, 10000, xor_dataset);
            brain.Save("xor_brain.json");

            Console.WriteLine("0 xor 0 => " + brain.Feedforward(Vector<double>.Build.DenseOfArray([0, 0]))[0]);
            Console.WriteLine("0 xor 1 => " + brain.Feedforward(Vector<double>.Build.DenseOfArray([0, 1]))[0]);
            Console.WriteLine("1 xor 0 => " + brain.Feedforward(Vector<double>.Build.DenseOfArray([1, 0]))[0]);
            Console.WriteLine("1 xor 1 => " + brain.Feedforward(Vector<double>.Build.DenseOfArray([1, 1]))[0]);
        }
    }
}